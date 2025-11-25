# Copyright 2025 the KVCache.AI team, Approaching AI, and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from transformers import EarlyStoppingCallback, PreTrainedModel, DataCollatorForSeq2Seq

from ..data import get_dataset, get_template_and_fix_tokenizer, SFTDataCollatorWith4DAttentionMask
from ..extras import logging
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_mcore_adapter_available, is_ray_available
from ..hparams import get_infer_args, get_ray_args, get_train_args, read_args
from ..model import load_model, load_tokenizer
from .callbacks import LogCallback, PissaConvertCallback, ReporterCallback
from .dpo import run_dpo
from .kto import run_kto
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft
from .trainer_utils import get_ray_trainer, get_swanlab_callback


if is_ray_available():
    import ray
    from ray.train.huggingface.transformers import RayTrainReportCallback


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



def run_attribution(model_args, data_args, training_args, finetuning_args, callbacks):
    logger.info("开始获取 Input Embedding 梯度...")

    # ================= 插入开始：激活 Liger Kernel =================
    if model_args.enable_liger_kernel:
        logger.info(f"检测到 enable_liger_kernel=True,正在为 attrib 阶段强制注入 Liger Kernel...")
        try:
            # 尝试 Qwen3
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3
            apply_liger_kernel_to_qwen3()
            logger.info("成功注入 Liger Kernel (Qwen3 模式)")
                
        except Exception as e:
            logger.warning(f"Liger Kernel 注入失败: {e}")
            logger.warning("将使用原生 PyTorch 算子继续运行。")

    # 基础组件加载
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 加载模型 
    model = load_model(tokenizer, model_args, finetuning_args)
    model.train()  # 确保模型在训练模式下

    if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False # 训练/梯度分析时必须关闭 KV Cache
            logger.info("已手动开启 Gradient Checkpointing (节省显存模式)")

    # 大部分 HF 模型用 get_input_embeddings() 都能拿到这一层
    embedding_layer = model.get_input_embeddings()
    # 准备数据迭代器
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    #  明确提取"训练集"
    # LLaMA-Factory 的 get_dataset 返回的字典里，训练集的 key 固定叫 "train_dataset"
    if "train_dataset" in dataset_module:
        dataset = dataset_module["train_dataset"]
    else:
        raise ValueError("未找到训练数据集 (train_dataset)，请检查数据配置。")

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    assert training_args.per_device_train_batch_size ==1, "当前仅支持 batch_size=1 的梯度分析。"
    # 现在的 dataset 就是真正的 Dataset 对象了，可以传给 DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1,  # bs ==1,分析单个batchsize
        collate_fn=data_collator
    )
    # 将梯度存储到本地磁盘，分析与获取分离
    cache_dir = os.path.join(training_args.output_dir, "grad_caches")
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"梯度缓存文件将保存至: {cache_dir}")

    for step, batch in enumerate(dataloader):
        # 将数据移到 GPU
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        # 清空梯度
        model.zero_grad()

        # 把 input_ids 单独拿出来，并从 batch 中移除
        if "input_ids" not in batch:
            raise ValueError("Batch 中未找到 'input_ids'，无法获取 Embedding 输入。")
        # 保存一份 input_ids 用于后续还原文本
        # batch["input_ids"] 在 pop 之后就没了，所以要先存下来
        original_input_ids = batch["input_ids"].clone()
        input_ids = batch.pop("input_ids") 
        
        # 手动通过 Embedding 层获取向量
        # 这一步只是查表，得到的是 Tensor
        inputs_embeds = embedding_layer(input_ids)
        
        # .detach(): 我们不关心 Embedding 表本身的权重怎么变
        # .requires_grad_(True): 我们只关心这个“输入向量”本身怎么变
        inputs_embeds = inputs_embeds.detach()
        inputs_embeds.requires_grad_(True)
        
        # 把处理好的 inputs_embeds 放回 batch
        batch["inputs_embeds"] = inputs_embeds
        
        # 此时模型接收的是 embeddings，跳过了内部的 embedding_layer 查表过程
        outputs = model(**batch)
        
        loss = outputs.loss 
        
        # 反向传播
        loss.backward()

        # 直接获取梯度
        # 因为我们设置了 inputs_embeds.requires_grad_(True)，梯度会直接存要在 inputs_embeds.grad 里
        grad_x = inputs_embeds.grad
        
        if grad_x is not None:
            logger.info(f"Step {step}: 成功获取梯度, Shape: {grad_x.shape}")
            
            # === 分析算法 ===
            grad_matrix = grad_x.detach().squeeze(0).float() # 转换成float32，保证svd计算精度
            full_input_ids = original_input_ids[0]
            full_labels = batch["labels"][0]
            # 寻找response的起点

            valid_label_indices = (full_labels !=-100).nonzero(as_tuple=True)[0]
            if (len(valid_label_indices) ==0):
                raise ValueError("当前样本未找到有效的 labels,无法定位 response 部分。")
            response_start_idx = valid_label_indices[0].item()
            # 只分析 response 部分的梯度结构
            grad_matrix = grad_matrix[response_start_idx:, :]  # 形状 [Resp_Len, Emb_Dim]
            logger.info(f"全长: {len(full_input_ids)}, Prompt长度: {response_start_idx}, Response长度: {grad_matrix.shape[0]}")

            save_data = {
                "step": step,
                "input_ids": original_input_ids[0].cpu().tolist(),
                "tokens": tokenizer.convert_ids_to_tokens(original_input_ids[0]), # 预先转好 Token 字符串方便后续使用
                "gradients": grad_x.detach().squeeze(0).cpu(), # [Seq_Len, Hidden_Dim]
                "response_start_idx": response_start_idx,
                "loss": loss.item()
            }

            # 3. 写入文件
            save_path = os.path.join(cache_dir, f"sample_{step}.pt")
            torch.save(save_data, save_path)
            logger.info(f"Step {step}: 梯度已保存至 {save_path}")

        else:
            logger.error("梯度依然为空！")

    logger.info("梯度获取结束。")

def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    if finetuning_args.early_stopping_steps is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=finetuning_args.early_stopping_steps))

    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last

    if finetuning_args.stage in ["pt", "sft", "dpo"] and finetuning_args.use_mca:
        if not is_mcore_adapter_available():
            raise ImportError("mcore_adapter is not installed. Please install it with `pip install mcore-adapter`.")
        if finetuning_args.stage == "pt":
            from .mca import run_pt as run_pt_mca

            run_pt_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "sft":
            from .mca import run_sft as run_sft_mca

            run_sft_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "dpo":
            from .mca import run_dpo as run_dpo_mca

            run_dpo_mca(model_args, data_args, training_args, finetuning_args, callbacks)

    elif finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        if model_args.use_kt:
            from .ksft.workflow import run_sft as run_sft_kt

            run_sft_kt(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
        else:
            run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)

    elif finetuning_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "kto":
        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "attrib":
        run_attribution(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError(f"Unknown task: {finetuning_args.stage}.")

    if is_ray_available() and ray.is_initialized():
        return  # if ray is intialized it will destroy the process group on return

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")


def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    ray_args = get_ray_args(args)
    callbacks = callbacks or []
    if ray_args.use_ray:
        callbacks.append(RayTrainReportCallback())
        trainer = get_ray_trainer(
            training_function=_training_function,
            train_loop_config={"args": args, "callbacks": callbacks},
            ray_args=ray_args,
        )
        trainer.fit()
    else:
        _training_function(config={"args": args, "callbacks": callbacks})


def export_model(args: Optional[dict[str, Any]] = None) -> None:
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    if getattr(model, "quantization_method", None) is not None and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is not None:  # quantized model adopts float16 type
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        if model_args.infer_dtype == "auto":
            output_dtype = getattr(model.config, "torch_dtype", torch.float32)
            if output_dtype == torch.float32:  # if infer_dtype is auto, try using half precision first
                output_dtype = infer_optim_dtype(torch.bfloat16)
        else:
            output_dtype = getattr(torch, model_args.infer_dtype)

        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info_rank0(f"Convert model dtype to: {output_dtype}.")

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size=f"{model_args.export_size}GB",
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size=f"{model_args.export_size}GB",
            safe_serialization=(not model_args.export_legacy_format),
        )

    if finetuning_args.stage == "rm":
        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        if os.path.exists(os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_SAFE_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")
        elif os.path.exists(os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if processor is not None:
            processor.save_pretrained(model_args.export_dir)
            if model_args.export_hub_model_id is not None:
                processor.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

    except Exception as e:
        logger.warning_rank0(f"Cannot save tokenizer, please copy the files manually: {e}.")

    ollama_modelfile = os.path.join(model_args.export_dir, "Modelfile")
    with open(ollama_modelfile, "w", encoding="utf-8") as f:
        f.write(template.get_ollama_modelfile(tokenizer))
        logger.info_rank0(f"Ollama modelfile saved in {ollama_modelfile}")
