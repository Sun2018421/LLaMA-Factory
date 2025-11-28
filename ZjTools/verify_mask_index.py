#!/usr/bin/env python3
"""
验证mask的token index是否与SFT训练时的数据对应
模拟SFT训练时的数据处理流程，检查mask索引映射是否正确
"""
import os
import sys
import torch
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from transformers import AutoTokenizer, TrainingArguments

# 延迟导入，避免导入时的依赖问题
def _import_llamafactory_modules():
    """延迟导入llamafactory模块，避免导入时的依赖问题"""
    try:
        from llamafactory.data.loader import _get_merged_dataset, _get_preprocessed_dataset
        from llamafactory.hparams import DataArguments, ModelArguments
        from llamafactory.data.template import get_template_and_fix_tokenizer
        from llamafactory.extras.constants import IGNORE_INDEX
        return _get_merged_dataset, _get_preprocessed_dataset, DataArguments, ModelArguments, get_template_and_fix_tokenizer, IGNORE_INDEX
    except ImportError as e:
        # 如果导入失败，尝试修复mm_plugin的导入问题
        if 'mllama' in str(e):
            # 临时创建一个假的mllama模块
            import transformers.models
            if not hasattr(transformers.models, 'mllama'):
                class FakeMllama:
                    class processing_mllama:
                        pass
                transformers.models.mllama = FakeMllama()
            # 重新导入
            from llamafactory.data.loader import _get_merged_dataset, _get_preprocessed_dataset
            from llamafactory.hparams import DataArguments, ModelArguments
            from llamafactory.data.template import get_template_and_fix_tokenizer
            from llamafactory.extras.constants import IGNORE_INDEX
            return _get_merged_dataset, _get_preprocessed_dataset, DataArguments, ModelArguments, get_template_and_fix_tokenizer, IGNORE_INDEX
        raise


# 在函数中使用时导入
_get_merged_dataset = None
_get_preprocessed_dataset = None
DataArguments = None
ModelArguments = None
get_template_and_fix_tokenizer = None
IGNORE_INDEX = -100  # 默认值

def load_mask_file(mask_dir, sample_idx):
    """加载mask文件"""
    mask_file = Path(mask_dir) / f"sample_{sample_idx}_mask.pt"
    if not mask_file.exists():
        return None
    return torch.load(mask_file, map_location="cpu")


def simulate_sft_preprocessing(data_args, template, tokenizer, sample_idx, model_path, apply_mask=False, mask_dir=None):
    """模拟SFT训练时的数据预处理流程
    
    Args:
        apply_mask: 是否应用mask（用于验证mask应用后的结果）
        mask_dir: mask文件目录（如果apply_mask=True）
        model_path: 模型路径（用于ModelArguments）
    """
    # 创建简单的training_args用于数据加载
    training_args = TrainingArguments(
        output_dir="./tmp",
        per_device_train_batch_size=1,
        remove_unused_columns=False,
    )
    # 设置predict_with_generate属性（如果不存在）
    if not hasattr(training_args, 'predict_with_generate'):
        training_args.predict_with_generate = False
    
    model_args = ModelArguments(model_name_or_path=model_path)
    
    # 获取合并后的数据集
    dataset = _get_merged_dataset(
        data_args.dataset,
        model_args,
        data_args,
        training_args,
        stage="sft",
    )
    
    if dataset is None or len(dataset) == 0:
        return None
    
    if sample_idx >= len(dataset):
        print(f"⚠️  样本索引 {sample_idx} 超出数据集大小 {len(dataset)}")
        return None
    
    # 临时移除attr_mask_dir，因为我们想手动控制mask应用
    original_attr_mask_dir = data_args.attr_mask_dir
    if not apply_mask:
        data_args.attr_mask_dir = None
    
    # 预处理数据集（这会应用tokenization和template）
    dataset = _get_preprocessed_dataset(
        dataset,
        data_args,
        training_args,
        stage="sft",
        template=template,
        tokenizer=tokenizer,
        processor=None,
        is_eval=False,
    )
    
    # 恢复attr_mask_dir
    data_args.attr_mask_dir = original_attr_mask_dir
    
    if dataset is None or len(dataset) == 0:
        return None
    
    if sample_idx >= len(dataset):
        print(f"⚠️  样本索引 {sample_idx} 超出预处理后数据集大小 {len(dataset)}")
        return None
    
    # 如果需要在预处理时应用mask（模拟实际训练流程）
    if apply_mask and mask_dir:
        from pathlib import Path
        mask_file = Path(mask_dir) / f"sample_{sample_idx}_mask.pt"
        if mask_file.exists():
            payload = torch.load(mask_file, map_location="cpu")
            example = dataset[sample_idx]
            labels = example["labels"]
            if isinstance(labels, list):
                labels = torch.tensor(labels)
            
            first_valid = next((i for i, v in enumerate(labels) if v != IGNORE_INDEX), None)
            resp_start = payload.get("response_start_idx", first_valid or 0)
            if first_valid is not None and first_valid != resp_start:
                resp_start = first_valid
            
            mask_tensor = payload.get("mask", None)
            if isinstance(mask_tensor, torch.Tensor):
                mask_tensor = mask_tensor.tolist()
            ignore_value = payload.get("ignore_value", -100)
            
            for rel, val in enumerate(mask_tensor):
                if val == ignore_value:
                    target = resp_start + rel
                    if 0 <= target < len(labels):
                        labels[target] = IGNORE_INDEX
                    else: 
                        raise ValueError(f"Mask索引越界: rel={rel}, target={target}, labels长度={len(labels)}")
            example["labels"] = labels.tolist() if isinstance(labels, torch.Tensor) else labels
            return example
    
    # 获取指定样本（不应用mask）
    example = dataset[sample_idx]
    
    return example


def verify_mask_application(example, mask_data, sample_idx, verbose=True):
    """验证mask应用是否正确"""
    if example is None or mask_data is None:
        return False
    
    labels = example["labels"]
    if isinstance(labels, list):
        labels = torch.tensor(labels)
    
    # 获取实际的response起始位置（第一个非IGNORE_INDEX的位置）
    first_valid = next((i for i, v in enumerate(labels) if v != IGNORE_INDEX), None)
    
    # 从mask文件获取的信息
    mask_resp_start = mask_data.get("response_start_idx", None)
    mask_tensor = mask_data.get("mask", None)
    ignore_value = mask_data.get("ignore_value", -100)
    mask_response_length = mask_data.get("response_length", None)
    
    if isinstance(mask_tensor, torch.Tensor):
        mask_tensor = mask_tensor.tolist()
    
    # 计算实际的response长度
    if first_valid is not None:
        actual_resp_start = first_valid
        actual_resp_length = (labels != IGNORE_INDEX).sum().item() #通过原先labels中不是mask的数量来计算的
    else:
        actual_resp_start = 0
        actual_resp_length = 0
    
    # 验证信息
    results = {
        "sample_idx": sample_idx,
        "mask_resp_start": mask_resp_start,
        "actual_resp_start": actual_resp_start,
        "mask_resp_length": mask_response_length,
        "actual_resp_length": actual_resp_length,
        "mask_tensor_length": len(mask_tensor),
        "labels_length": len(labels),
        "match": True,
        "issues": []
    }
    
    # 检查1: response起始位置是否匹配
    if mask_resp_start is not None:
        if first_valid is not None and first_valid != mask_resp_start:
            results["match"] = False
            results["issues"].append(
                f"Response起始位置不匹配: mask中={mask_resp_start}, 实际={first_valid}"
            )
            # 使用实际的起始位置（代码中的逻辑）
            resp_start = first_valid
        else:
            resp_start = mask_resp_start
    else:
        resp_start = first_valid or 0
    
    # 检查2: mask tensor长度是否与response长度匹配
    if mask_response_length is not None:
        if mask_response_length != len(mask_tensor):
            results["match"] = False
            results["issues"].append(
                f"Mask tensor长度与response_length不匹配: "
                f"mask_response_length={mask_response_length}, mask_tensor长度={len(mask_tensor)}"
            )
    
    if actual_resp_length > 0 and len(mask_tensor) != actual_resp_length:
        results["match"] = False
        results["issues"].append(
            f"Mask tensor长度与实际response长度不匹配: "
            f"mask_tensor长度={len(mask_tensor)}, 实际response长度={actual_resp_length}"
        )
    
    # 检查3: 模拟mask应用过程
    masked_positions = []
    for rel, val in enumerate(mask_tensor):
        if val == ignore_value:
            target = resp_start + rel
            if 0 <= target < len(labels):
                masked_positions.append(target)
            else:
                results["match"] = False
                results["issues"].append(
                    f"Mask索引越界: rel={rel}, target={target}, labels长度={len(labels)}"
                )
    
    results["masked_count"] = len(masked_positions)
    results["masked_positions"] = masked_positions[:10]  # 只保存前10个用于显示
    
    # 检查4: 验证mask后的labels
    if verbose:
        print(f"\n{'='*60}")
        print(f"样本 {sample_idx} 的Mask验证结果")
        print(f"{'='*60}")
        print(f"📋 基本信息:")
        print(f"   - Labels总长度: {len(labels)}")
        print(f"   - Mask中的response_start_idx: {mask_resp_start}")
        print(f"   - 实际的response起始位置: {actual_resp_start}")
        print(f"   - 使用的response起始位置: {resp_start}")
        print(f"   - Mask中的response_length: {mask_response_length}")
        print(f"   - Mask tensor长度: {len(mask_tensor)}")
        print(f"   - 实际response长度: {actual_resp_length}")
        
        print(f"\n🎯 Mask应用检查:")
        print(f"   - 被mask的token数量: {len(masked_positions)}")
        if len(masked_positions) > 0:
            print(f"   - 前10个被mask的位置: {masked_positions[:10]}")
        
        if results["issues"]:
            print(f"\n发现的问题:")
            for issue in results["issues"]:
                print(f"   - {issue}")
        else:
            print(f"\n所有检查通过！")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="验证mask的token index是否与SFT训练数据对应")
    parser.add_argument("--sample_idx", type=int, default=0, help="要验证的样本索引")
    parser.add_argument("--mask_dir", type=str,
                       default="saves/qwen3-4b/full/attr_temp1.0/masks",
                       help="Mask文件目录")
    parser.add_argument("--model_path", type=str,
                       default="/mnt/zj-gpfs/home/whs/model/Qwen3-4B-Instruct-2507",
                       help="模型路径")
    parser.add_argument("--dataset", type=str, default="math_cot",
                       help="数据集名称")
    parser.add_argument("--template", type=str, default="qwen",
                       help="模板名称")
    parser.add_argument("--cutoff_len", type=int, default=18000,
                       help="截断长度")
    parser.add_argument("--packing", action="store_true",
                       help="是否使用packing")
    # seed参数在TrainingArguments中，这里不需要
    parser.add_argument("--batch_samples", type=int, nargs="+", default=None,
                       help="批量验证多个样本，例如: --batch_samples 0 10 50 100")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Mask索引验证工具")
    print("=" * 60)
    
    # 导入llamafactory模块
    global _get_merged_dataset, _get_preprocessed_dataset, DataArguments, ModelArguments, get_template_and_fix_tokenizer, IGNORE_INDEX
    try:
        _get_merged_dataset, _get_preprocessed_dataset, DataArguments, ModelArguments, get_template_and_fix_tokenizer, IGNORE_INDEX = _import_llamafactory_modules()
    except Exception as e:
        print(f"❌ 导入llamafactory模块失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 准备数据参数（需要先创建data_args，因为template需要它）
    print(f"\n[1/3] 准备数据参数")
    data_args = DataArguments(
        dataset=args.dataset,
        template=args.template,
        cutoff_len=args.cutoff_len,
        packing=args.packing,
        overwrite_cache=True,
        preprocessing_num_workers=1,  # 验证时使用单进程
    )
    
    # 加载tokenizer和template
    print(f"\n[2/3] 加载模型和tokenizer: {args.model_path}")
    try:
        # 直接使用AutoTokenizer加载，避免复杂的导入
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        template = get_template_and_fix_tokenizer(
            tokenizer=tokenizer,
            data_args=data_args
        )
        print("✅ Tokenizer和template加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    print(f"   - 数据集: {args.dataset}")
    print(f"   - 模板: {args.template}")
    print(f"   - 截断长度: {args.cutoff_len}")
    print(f"   - Packing: {args.packing}")
    
    # 确定要验证的样本列表
    if args.batch_samples:
        sample_indices = args.batch_samples
    else:
        sample_indices = [args.sample_idx]
    
    print(f"\n[3/4] 验证样本: {sample_indices}")
    
    all_results = []
    for sample_idx in sample_indices:
        print(f"\n{'='*60}")
        print(f"验证样本 {sample_idx}")
        print(f"{'='*60}")
        
        # 加载mask文件
        mask_data = load_mask_file(args.mask_dir, sample_idx)
        if mask_data is None:
            print(f"❌ 无法加载mask文件: {args.mask_dir}/sample_{sample_idx}_mask.pt")
            continue
        
        # 模拟SFT预处理（不应用mask，用于验证mask索引）
        try:
            example = simulate_sft_preprocessing(
                data_args, template, tokenizer, sample_idx, args.model_path,
                apply_mask=False, mask_dir=None
            )
            if example is None:
                print(f"❌ 无法获取样本 {sample_idx}")
                continue
        except Exception as e:
            print(f"❌ 预处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 也获取应用mask后的样本用于对比
        try:
            example_with_mask = simulate_sft_preprocessing(
                data_args, template, tokenizer, sample_idx, args.model_path,
                apply_mask=True, mask_dir=args.mask_dir
            )
        except Exception as e:
            example_with_mask = None
            print(f"⚠️  无法获取应用mask后的样本: {e}")
        
        # 验证mask应用
        result = verify_mask_application(example, mask_data, sample_idx, verbose=True)
        
        # 如果成功获取了应用mask后的样本，进行额外验证
        if example_with_mask is not None:
            labels_before = example["labels"]
            labels_after = example_with_mask["labels"]
            if isinstance(labels_before, list):
                labels_before = torch.tensor(labels_before)
            if isinstance(labels_after, list):
                labels_after = torch.tensor(labels_after)
            
            # 检查mask是否正确应用
            before_ignore_count = (labels_before == IGNORE_INDEX).sum().item()
            after_ignore_count = (labels_after == IGNORE_INDEX).sum().item()
            additional_masked = after_ignore_count - before_ignore_count
            
            print(f"对比应用mask前后:")
            print(f"   - 应用mask前IGNORE_INDEX数量: {before_ignore_count}")
            print(f"   - 应用mask后IGNORE_INDEX数量: {after_ignore_count}")
            print(f"   - 新增mask的token数量: {additional_masked}")
            print(f"   - Mask文件中的masked数量: {result.get('masked_count', 0)}")
            
            if additional_masked != result.get('masked_count', 0):
                result["match"] = False
                result["issues"].append(
                    f"Mask应用数量不匹配: 实际新增={additional_masked}, "
                    f"预期={result.get('masked_count', 0)}"
                )
        
        all_results.append(result)
    
    # 总结
    print(f"\n{'='*60}")
    print("验证总结")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in all_results if r and r.get("match", False))
    total_count = len(all_results)
    
    print(f"总样本数: {total_count}")
    print(f"验证通过: {success_count}")
    print(f"验证失败: {total_count - success_count}")
    
    if success_count < total_count:
        print(f"\n⚠️  部分样本验证失败，请检查上述问题")
    else:
        print(f"\n✅ 所有样本验证通过！")


if __name__ == "__main__":
    main()

