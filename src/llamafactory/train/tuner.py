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

def generate_full_report(input_ids, scores, tokenizer, step=0, save_dir="analysis_reports"):
    """
    功能：
    1. 画一张超宽的折线图 (2000个点也能看清)
    2. 生成一个交互式 HTML 网页，可以直接阅读带颜色的文本
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 预处理：转 List
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().tolist()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    seq_len = len(input_ids)
    
    # ==========================================
    # 1. 宏观视角：超宽折线图 (The Panorama)
    # ==========================================
    # 宽度设为 30 英寸，确保 2000 个点不会挤在一起
    plt.figure(figsize=(30, 6)) 
    
    plt.plot(scores, color='#8e44ad', linewidth=1.5, alpha=0.9, label='Importance (Leverage Score)')
    plt.fill_between(range(seq_len), scores, color='#8e44ad', alpha=0.1) # 填充颜色，更有质感
    
    # 标注均值线
    mean_score = np.mean(scores)
    plt.axhline(y=mean_score, color='red', linestyle='--', alpha=0.5, label=f'Average: {mean_score:.4f}')
    
    plt.title(f"Step {step} - Full Token Importance Panorama (Length: {seq_len})", fontsize=16)
    plt.xlabel("Token Position", fontsize=12)
    plt.ylabel("Leverage Score", fontsize=12)
    plt.legend(loc='upper right')
    plt.xlim(0, seq_len)
    plt.grid(True, alpha=0.2)
    
    # 保存图片
    img_path = os.path.join(save_dir, f"step_{step}_panorama.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150) # 150 dpi 保证清晰度
    plt.close()
    print(f"[1/2] 全景折线图已保存: {img_path}")

    # ==========================================
    # 微观视角：交互式 HTML 热力图 (The Heatmap)
    # ==========================================
    
    # 归一化分数 (Log Scale 增强对比度)
    # 加上 1e-10 避免 log(0)
    log_scores = np.log(scores + 1e-10)
    norm_scores = (log_scores - np.min(log_scores)) / (np.max(log_scores) - np.min(log_scores))
    
    # 准备 HTML 头部
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Token Attribution Analysis - Step {step}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; padding: 20px; background-color: #f9f9f9; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }}
            h2 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .token-box {{ 
                display: inline-block; 
                padding: 2px 4px; 
                margin: 0 1px; 
                border-radius: 4px; 
                transition: all 0.2s;
                cursor: default;
                white-space: pre-wrap; /* 【关键修改】保留原始的换行符和空格 */
                vertical-align: bottom; /* 保持对齐美观 */
            }}
            .token-box:hover {{ 
                transform: scale(1.1); 
                box-shadow: 0 2px 5px rgba(0,0,0,0.2); 
                z-index: 10;
                position: relative;
                border: 1px solid #333;
            }}
            .legend {{ margin-bottom: 20px; padding: 10px; background: #eee; border-radius: 4px; font-size: 0.9em; }}
            .stat-box {{ float: right; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Step {step} - Token Importance Heatmap</h2>
            <div class="legend">
                <span class="stat-box">Total Tokens: {seq_len}</span>
                <strong>Color Scale:</strong> 
                <span style="background-color: #fff0f0; padding: 2px 5px;">Low Info (Waste)</span> &rarr; 
                <span style="background-color: #ffcccc; padding: 2px 5px;">Medium</span> &rarr; 
                <span style="background-color: #ff0000; color: white; padding: 2px 5px;">High Info (Key)</span>
                <br>
                <small>* Hover over words to see raw leverage scores.</small>
            </div>
            <div style="font-size: 16px; word-wrap: break-word;">
    """

    # 逐个生成 Token 的 HTML
    cmap = plt.get_cmap('Reds') # 使用红色系
    
    for i, (token_id, score, raw_score) in enumerate(zip(input_ids, norm_scores, scores)):
        token_text = tokenizer.decode([token_id])
        if "\ufffd" in token_text:
        # 如果解码失败，说明这是个“半截”字符
        # 我们改用 convert_ids_to_tokens 获取原始字节表示
            raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]
        
            # Qwen 的 raw_token 通常是 bytes 类型，转成字符串显示
            if isinstance(raw_token, bytes):
                # 显示为 <0xE2> 这种格式，既专业又不会乱码
                display_text = f"&lt;0x{raw_token.hex().upper()}&gt;"
            else:
                # 如果不是 bytes，就保留原样或做简单转义
                display_text = str(raw_token).replace('<', '&lt;').replace('>', '&gt;')
        else:
            # 如果解码正常，就用正常的文本
            display_text = token_text.replace('<', '&lt;').replace('>', '&gt;')
        
        # 获取颜色 (分数越低越白，越高越红)
        # 我们可以设置一个阈值，如果分数极低直接全白，减少视觉干扰
        if raw_score < np.mean(scores) * 0.5:
            color_hex = "#ffffff" # 纯白
            text_color = "#ccc"   # 浅灰字
        else:
            rgba = cmap(score) # 0-1 映射到颜色
            color_hex = mcolors.to_hex(rgba)
            text_color = "#000" if score < 0.7 else "#fff" # 深背景用白字

        html_content += f"""
    <span class="token-box" 
            style="background-color: {color_hex}; color: {text_color};" 
            title="idx: {i} | token: {repr(token_text)} | score: {raw_score:.6f}">{display_text}</span>"""

    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    # 保存 HTML
    html_path = os.path.join(save_dir, f"step_{step}_heatmap.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[2/2] 交互式网页已保存: {html_path}")
    print(f"请使用浏览器打开 {html_path} 进行观察。")

# =================使用示例=================
# 在你的主循环里这样调用：
# generate_full_report(current_input_ids, token_importance_scores, tokenizer, step=step)

def get_color_hex(score, cmap_name='Reds'):
    """将归一化后的分数转换为 HTML 颜色代码"""
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(score)
    return mcolors.to_hex(rgba)

def create_colored_visualization(input_ids, scores, tokenizer):
    """生成带有颜色高亮的 HTML 字符串"""
    # 1. Log 归一化 (关键优化：让低分区域的层次更明显)
    # 加 1e-10 防止 log(0)
    log_scores = np.log(scores + 1e-10)
    min_val = np.min(log_scores)
    max_val = np.max(log_scores)
    
    # 归一化到 0-1
    if max_val - min_val == 0:
        norm_scores = np.zeros_like(scores)
    else:
        norm_scores = (log_scores - min_val) / (max_val - min_val)

    html_parts = []
    plain_text_parts = []

    # 转回 list 方便遍历
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().tolist()
        
    for token_id, score, raw_score in zip(input_ids, norm_scores, scores):
        token_text = tokenizer.decode([token_id])
        
        # 转义 HTML 特殊字符
        safe_text = token_text.replace('<', '&lt;').replace('>', '&gt;')
        
        # 获取颜色
        color_hex = get_color_hex(score)
        
        # 构造 span 标签，title 里放原始分数，鼠标悬停可以看到
        span = f'<span style="background-color: {color_hex}; padding: 0 1px; border-radius: 2px;" title="Raw Score: {raw_score:.4f}"> {safe_text} </span>'
        
        html_parts.append(span)
        plain_text_parts.append(token_text)

    return "".join(html_parts), "".join(plain_text_parts)

def run_attribution(model_args, data_args, training_args, finetuning_args, callbacks):
    logger.info("开始获取 Input Embedding 梯度...")

    # 基础组件加载
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 加载模型 
    model = load_model(tokenizer, model_args, finetuning_args)
    model.train()  # 确保模型在训练模式下

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
            
            U, S, V = torch.svd(grad_matrix)
            singular_values = S.cpu().numpy() 

            # 绘制图片
            # 计算累积能量
            energy = np.cumsum(singular_values ** 2) / np.sum(singular_values ** 2)
            # 计算 Leverage Score ---
            # 自动寻找 95% 能量的截断点 k
            threshold = 0.95
            k_indices = np.where(energy >= threshold)[0]
            k = k_indices[0] + 1 if len(k_indices) > 0 else len(energy)
            print(f"解释 {threshold*100}% 能量所需的秩 k = {k}")
            # 只取前 k 个左奇异向量 (U_k)
            U_k = U[:, :k]  # 形状 [Seq_Len, k]
            # 计算每一行(Token)的 L2 范数平方 -> 在k维度中，根据投影来判断重要性分数
            token_importance_scores = torch.norm(U_k, dim=1).pow(2).cpu().numpy()
            full_scores = np.zeros(len(full_input_ids))
            full_scores[response_start_idx:] = token_importance_scores
            # ==========================================
            # 组合绘图 (三图合一)
            # ==========================================
            plt.figure(figsize=(18, 5))
            # 图 1: 奇异值分布 (The Cliff)
            plt.subplot(1, 3, 1)
            plt.plot(singular_values[:k], marker='o', markersize=3) # 只看前200个细节
            plt.title(f"Top k Singular Values\n(Low Rank Structure)", fontsize=12)
            plt.xlabel("Rank Index")
            plt.ylabel("Singular Value")
            plt.grid(True, alpha=0.3)
            # 图 2: 能量累积 (Cumulative Energy)
            plt.subplot(1, 3, 2)
            plt.plot(energy[:k], color='orange', linewidth=2)
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold*100}% Energy')
            plt.axvline(x=k, color='g', linestyle='--', label=f'Rank k={k}')
            plt.title(f"Cumulative Energy Explained\n(k={k} captures {threshold*100}%)", fontsize=12)
            plt.xlabel("Rank Index")
            plt.ylabel("Ratio")
            plt.legend()
            plt.grid(True, alpha=0.3)
            # 图 3: Token 重要性分布 (Token-wise Leverage Scores)
            # 这是你要找"废话"的关键图
            plt.subplot(1, 3, 3)
            plt.plot(token_importance_scores, color='purple', linewidth=1, alpha=0.8)
            # 可以画个均值线参考
            mean_score = np.mean(token_importance_scores)
            plt.axhline(y=mean_score, color='black', linestyle=':', label='Avg Importance')

            plt.title("Token Leverage Scores\n(High=Important, Low=Waste)", fontsize=12)
            plt.xlabel("Token Position (Sequence)")
            plt.ylabel("Importance Score (Leverage)")
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"leverage_scores_analysis_{step}sample.png")

            ## 绘制report 
            generate_full_report(full_input_ids, full_scores, tokenizer, step=step, save_dir="attribution_reports")
        else:
            logger.error("梯度依然为空！")

        # 示例只跑第一个sample，看看效果
        if step >= 0: 
            break

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
