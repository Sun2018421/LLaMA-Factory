import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from tqdm import tqdm
import re


class segmentAnalyzer:
    '''
    segmentAnalyzer 的 Docstring
    用来基于正则表达式将文本切分为多个段落/句子，并计算每个段落的归因强度。
    '''
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer 
        # 用户指定的严格正则
        # Group 1: 结束标点 (. ? ! } ])
        # Group 2: 空白或换行
        # Group 3: 下一句的大写开头字母
        self.split_pattern = re.compile(r'([.?!}\]])([\s\n]+)([A-Z])')
    
    def get_segments(self, input_ids):
        """
        基于正则表达式进行切分。
        逻辑：
        1. 将 Token ID 解码为完整字符串。
        2. 建立 字符索引 -> Token 索引 的映射表。
        3. 用正则寻找切分点（句子边界）。
        4. 将切分点的字符位置映射回 Token 位置。
        """
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().tolist()
            
        # 1. 解码并建立映射 (Char Index -> Token Index)
        # 我们需要知道字符串里的第N个字符，对应的是第几个Token
        full_text = ""
        char_to_token_idx = []
        
        for i, tid in enumerate(input_ids):
            # 解码单个 token
            s = self.tokenizer.decode([tid])
            full_text += s
            # 将当前 token 的索引 i，填入对应长度的映射表中
            char_to_token_idx.extend([i] * len(s))
            
        segments = []
        current_token_start = 0
        
        # 2. 使用正则查找所有匹配项
        # finditer 会返回所有匹配对象
        for match in self.split_pattern.finditer(full_text):
            # 我们希望在新句子的开头切分
            # match.group(3) 是匹配到的大写字母
            # match.start(3) 是该大写字母在 full_text 中的字符索引
            split_char_index = match.start(3)
            
            # 边界检查
            if split_char_index < len(char_to_token_idx):
                # 找到该字符对应的 Token 索引
                split_token_idx = char_to_token_idx[split_char_index]
                
                # 只有当切分点确实在推移（避免重复切分）且不是第一段时
                if split_token_idx > current_token_start:
                    segments.append((current_token_start, split_token_idx))
                    current_token_start = split_token_idx
        
        # 3. 加入最后一段
        if current_token_start < len(input_ids):
            segments.append((current_token_start, len(input_ids)))
            
        return segments

    def analyze(self, input_ids, token_scores):
        """
        计算 Segment-level Metrics 
        input_ids: token id 列表
        token_scores: SVD 计算出的 Importance Score
        """
        segments = self.get_segments(input_ids)
        segment_metrics = []

        for start, end in segments:
            length = end - start
            if length == 0: continue

            # 获取该段的 Token Scores (Magnitude)
            seg_scores = token_scores[start:end]
            
            # 计算 Attribution Strength
            # Strength = Sum( token scores) / sqrt(N)
            strength = np.sum(seg_scores) / np.sqrt(length)

            segment_metrics.append({
                "start": start,
                "end": end,
                "text": self.tokenizer.decode(input_ids[start:end]),
                "strength": strength,
                "token_scores": seg_scores,
                "tokens_str": [self.tokenizer.decode([tid]) for tid in input_ids[start:end]]
            })
            
        return segment_metrics


def generate_segment_report(segment_data, prompt_text, sample_idx=0, save_dir="analysis_reports"):
    """
    带 Top/Bottom 标记的极简版报告：
    1. 预计算所有片段的排名。
    2. Top 10: 深红底白字+加粗。
    3. Bottom 10: 浅灰底灰字+变淡。
    4. 其余: 维持热力图渐变。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # === 预处理：计算排名 (只针对非空文本) ===
    # 我们需要知道哪些索引属于 Top 10 和 Bottom 10
    indexed_segments = []
    for i, seg in enumerate(segment_data):
        # 过滤掉纯换行或纯空格，不参与排名
        if seg['text'].strip():
            indexed_segments.append((i, seg['strength']))
    
    # 按分数从高到低排序
    sorted_segs = sorted(indexed_segments, key=lambda x: x[1], reverse=True)
    total_ranked = len(sorted_segs)
    
    # 获取 Top 10
    top_10_indices = set(x[0] for x in sorted_segs[:10])
    
    # 计算后 10% / 20% / 30% 
    # 先按升序（最不重要在前）排列
    ascending_segs = list(reversed(sorted_segs))
    thr_10 = max(1, int(np.ceil(total_ranked * 0.1))) if total_ranked else 0
    thr_20 = max(thr_10, int(np.ceil(total_ranked * 0.2))) if total_ranked else 0
    thr_30 = max(thr_20, int(np.ceil(total_ranked * 0.3))) if total_ranked else 0
    
    bottom_10_pct = set(x[0] for x in ascending_segs[:thr_10])
    bottom_20_pct = set(x[0] for x in ascending_segs[thr_10:thr_20])
    bottom_30_pct = set(x[0] for x in ascending_segs[thr_20:thr_30])

    # === 计算颜色范围 (用于中间部分的渐变) ===
    all_strengths = [s['strength'] for s in segment_data]
    if all_strengths:
        max_str = max(all_strengths)
        min_str = min(all_strengths)
        range_str = max_str - min_str if max_str != min_str else 1.0
    else:
        max_str, min_str, range_str = 1.0, 0.0, 1.0

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Sample {sample_idx} - Top/Bottom Analysis</title>
        <style>
            body {{ 
                font-family: 'Times New Roman', serif; 
                font-size: 18px;
                line-height: 1.6;
                color: #000;
                background: #fff;
                margin: 40px auto;
                max-width: 900px;
            }}
            
            h2 {{ border-bottom: 2px solid #333; padding-bottom: 10px; margin-top: 40px; }}
            .prompt-box {{ background: #f4f4f4; padding: 15px; margin-bottom: 30px; font-family: sans-serif; font-size: 16px; white-space: pre-wrap; color: #555; }}
            .content-area {{ text-align: justify; }}

            /* 基础样式 */
            span.seg {{
                padding: 0 2px; /* 稍微加一点点左右间距，让色块分明一点点，也可设为0 */
                -webkit-box-decoration-break: clone;
                box-decoration-break: clone;
            }}
            
            /* 特殊标记样式 */
            .top-rank {{
                font-weight: bold;
                border-bottom: 2px solid #8B0000; /* 底部加粗线 */
            }}
            .bottom-rank {{
                color: #999 !important; /* 强制灰色字 */
                font-style: italic;
            }}
        </style>
    </head>
    <body>
        <h2>Sample {sample_idx} Analysis</h2>
        <div class="prompt-box"><strong>Context:</strong><br>{prompt_text}</div>
        
        <h2>Reasoning Trace (Top 10 vs Bottom 10)</h2>
        <div class="content-area">
    """

    for i, seg in enumerate(segment_data):
        strength = seg['strength']
        raw_text = seg['text']
        
        # 文本清洗：换行转空格
        text = raw_text.replace("\n", " ").replace("<|im_end|>", "")
        
        if not text.strip():
            continue # 跳过空块
            
        # === 核心逻辑：根据排名决定样式 ===
        if i in top_10_indices:
            # Top 10: 深红底，白字，加粗
            bg_color = "#8B0000" # 深红
            font_color = "#FFFFFF"
            css_class = "seg top-rank"
            tooltip = f"Rank: TOP 10 | Score: {strength:.4f}"
        elif i in bottom_10_pct:
            # 后 10%：最弱，深灰
            bg_color = "#C0C0C0"
            font_color = "#555555"
            css_class = "seg bottom-rank"
            tooltip = f"Rank: Bottom 10% | Score: {strength:.4f}"
        elif i in bottom_20_pct:
            # 10%-20%：中灰
            bg_color = "#D8D8D8"
            font_color = "#666666"
            css_class = "seg bottom-rank"
            tooltip = f"Rank: Bottom 20% | Score: {strength:.4f}"
        elif i in bottom_30_pct:
            # 20%-30%：浅灰
            bg_color = "#E8E8E8"
            font_color = "#777777"
            css_class = "seg bottom-rank"
            tooltip = f"Rank: Bottom 30% | Score: {strength:.4f}"
            
        else:
            # 其他中间部分：使用原本的 Heatmap 逻辑 (淡红 -> 中红)
            norm = (strength - min_str) / range_str
            # alpha 范围控制在 0.05 到 0.5 之间，避免抢了 Top 10 的风头
            alpha = 0.05 + (norm ** 2) * 0.5 
            bg_color = f"rgba(255, 0, 0, {alpha:.2f})"
            font_color = "#000"
            css_class = "seg"
            tooltip = f"Score: {strength:.4f}"

        # HTML 转义
        display_text = text.replace("<", "&lt;").replace(">", "&gt;")
        
        html_content += f'<span class="{css_class}" style="background-color: {bg_color}; color: {font_color};" title="{tooltip}">{display_text}</span>'
        
    html_content += """
        </div>
        <div style="margin-top: 30px; font-size: 14px; color: #666; border-top: 1px solid #ddd; padding-top: 10px;">
            <strong>Legend:</strong> 
            <span style="background:#8B0000; color:white; padding:0 4px; font-weight:bold;">Deep Red</span> = Top 10 Important | 
            <span style="background:#C0C0C0; color:#555; padding:0 4px; font-style:italic;">Dark Gray</span> = Bottom 10% | 
            <span style="background:#D8D8D8; color:#666; padding:0 4px; font-style:italic;">Mid Gray</span> = Bottom 20% | 
            <span style="background:#E8E8E8; color:#777; padding:0 4px; font-style:italic;">Light Gray</span> = Bottom 30% | 
            <span style="color:red;">Light Red</span> = Intermediate
        </div>
    </body>
    </html>
    """
    
    html_path = os.path.join(save_dir, f"sample_{sample_idx}_ranked.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"分级报告已生成: {html_path}")

def analyze_single_file(file_path, output_root, seg_analyzer=None):
    """处理单个 .pt 文件,对应着一个梯度"""
    print(f"正在分析: {file_path}")
    data = torch.load(file_path)
    
    step = data['step']
    grad_matrix = data['gradients'].float() # [Seq, Dim]
    response_start = data['response_start_idx']
    input_ids = data['input_ids'] 
    all_tokens = data['tokens']

    # === 数据切分 ===
    # Prompt 部分：用于显示上下文，不参与归因
    prompt_ids = input_ids[:response_start]
    # 解码 Prompt 文本，用于后续单独展示
    prompt_text = seg_analyzer.tokenizer.decode(prompt_ids, skip_special_tokens=False)
    
    # Response 部分：用于归因分析
    target_grad = grad_matrix[response_start:, :]
    target_input_ids = input_ids[response_start:]
    target_tokens = all_tokens[response_start:]

    # === SVD 计算 (只针对 Response) ===
    try:
        U, S, V = torch.svd(target_grad)
    except Exception as e:
        raise RuntimeError(f"SVD 计算失败: {e}")

    singular_values = S.numpy()
    
    # 计算秩 k (95% 能量)
    energy = np.cumsum(singular_values ** 2) / np.sum(singular_values ** 2)
    threshold = 0.95
    k_indices = np.where(energy >= threshold)[0]
    k = k_indices[0] + 1 if len(k_indices) > 0 else len(energy)
    
    # 计算 Leverage Scores
    U_k = U[:, :k]
    response_scores = torch.norm(U_k, dim=1).pow(2).numpy()
    
    # === 生成报告 ===
    report_dir = os.path.join(output_root, "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    segment_data = seg_analyzer.analyze(target_input_ids, response_scores)
        
    generate_segment_report(
        segment_data, 
        prompt_text,   
        sample_idx=step, 
        save_dir=report_dir
    )
    print(f"Sample {step} 分析完成 -> {report_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, help="存放 .pt 文件的文件夹路径",default="saves/qwen3-4b/full/attr_temp1.0/grad_caches")
    parser.add_argument("--output_dir", type=str, default="saves/qwen3-4b/full/attr_temp1.0", help="分析结果输出路径")
    parser.add_argument("--model_path", type=str, default='/mnt/zj-gpfs/home/whs/model/Qwen3-4B-Instruct-2507', help="模型路径，用于加载 Tokenizer")
    parser.add_argument("--file_mode", type=str, choices=["all", "single"], default="all", help="选择处理全部梯度文件或单个文件")
    parser.add_argument("--file_name", type=str, default=None, help="当选择单个文件时，指定 .pt 文件名或 step")
    args = parser.parse_args()

    if not os.path.exists(args.cache_dir):
        print(f"错误: 文件夹 {args.cache_dir} 不存在")
        exit()

    # 获取要处理的 .pt 文件
    if args.file_mode == "all":
        files = [f for f in os.listdir(args.cache_dir) if f.endswith(".pt")]
        files.sort(key=lambda x: int(x.split('_')[1].split('.')[0])) # 按 step 排序
        if not files:
            print("错误: 未找到任何 .pt 文件")
            exit()
    else:
        if not args.file_name:
            print("错误: file_mode=single 时必须提供 --file_name")
            exit()
        # 支持传入 step 编号或完整文件名
        candidate = args.file_name
        if not candidate.endswith(".pt"):
            candidate = f"{candidate}.pt"
        target_path = os.path.join(args.cache_dir, candidate)
        if not os.path.exists(target_path):
            print(f"错误: 文件 {target_path} 不存在")
            exit()
        files = [candidate]

    print(f"找到 {len(files)} 个梯度缓存文件，开始分析...")

    print(f"正在加载 Tokenizer: {args.model_path} ...")
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Tokenizer 加载失败: {e}")
        exit()

    seg_analyzer = segmentAnalyzer(tokenizer)
    for f in tqdm(files):
        file_path = os.path.join(args.cache_dir, f)
        analyze_single_file(file_path, args.output_dir,seg_analyzer=seg_analyzer)
 