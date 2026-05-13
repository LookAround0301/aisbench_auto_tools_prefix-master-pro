#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- 入参：
  --total               数据集总条数（>=1）
  --num-prefixes        总公共前缀数（>=1）
  --length              单条请求长度（token数，>=1，变长模式下为默认/回退长度）
  --prefix-ratio        前缀占比；可用百分比如 "50%" 或小数如 "0.5"
  --tokenizer-dir       tokenizer 路径
  --seed                随机种子（int）
- 产物：
  1) 公共前缀（GSM8K格式）：保存到 ./{seed}_{length}_{前缀占比%}_{tokenizer名}_prefix/ 目录下
     文件：prefix.jsonl  —— 共 num_prefixes 行，每行 {"question": <prefix_text>, "answer": ""}
  2) 完整数据集（GSM8K格式）：保存到 ./{seed}_{length}_{前缀占比%}_{tokenizer名}.jsonl
     文件：dataset.jsonl —— 共 total 行，每行 {"question": <full_text>, "answer": ""}

python gen_multi_prefix_dataset.py \
  --total 500 \
  --num-prefixes 10 \
  --length 4096 \
  --prefix-ratio 50% \
  --tokenizer-dir /mnt/weight/deepseek_r1 \
  --seed 202508167
"""

import argparse
import json
import math
import os
import random
import re
from typing import List, Tuple, Optional

from transformers import AutoTokenizer
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ALLOWED_RE = re.compile(r'^[A-Za-z0-9 ]+$')

# ------------------------- 基础工具（不抛异常，内部自愈） -------------------------

def ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass  # 极端情况下忽略

def basename_of(path: str) -> str:
    try:
        name = os.path.basename(os.path.normpath(path))
        return name or "tokenizer"
    except Exception:
        return "tokenizer"

def decode_ids(tokenizer, ids: List[int]) -> str:
    try:
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    except Exception:
        return ""

def encode_ids(tokenizer, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return []

def encode_len(tokenizer, text: str) -> int:
    return len(encode_ids(tokenizer, text))

def is_allowed_text(s: str) -> bool:
    if not s:
        return False
    if any(c in s for c in ("\n", "\r", "\t")):
        return False
    return ALLOWED_RE.fullmatch(s) is not None

def filter_allowed(s: str) -> str:
    # 仅保留英文字母/数字/空格；把换行制表转空格
    s2 = "".join(ch for ch in s if (ch.isalnum() or ch == " "))
    s2 = s2.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return s2

def build_safe_token_pools(tokenizer) -> Tuple[List[int], List[int], int]:
    """
    返回：(nospace_ids, space_ids, filler_id)
    - 从 vocab 中筛选"安全 token"（解码仅含 A-Za-z0-9 空格），剔除特殊/换行。
    - space_ids: 以空格开头的"可读安全" token
    - nospace_ids: 其它"可读安全" token
    - filler_id：兜底填充 id（优先取 space_ids[0]）
    """
    nospace_ids, space_ids = [], []
    try:
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    except Exception:
        special_ids = set()
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)

    for tid in range(vocab_size):
        if tid in special_ids:
            continue
        piece = decode_ids(tokenizer, [tid])
        if not piece:
            continue
        if not is_allowed_text(piece):
            continue
        if piece.startswith(" "):
            space_ids.append(tid)
        else:
            nospace_ids.append(tid)

    filler_id = None
    if space_ids:
        filler_id = space_ids[0]
    else:
        for cand in [" 0", " 1", " a", " A", " B", " 2"]:
            ids = encode_ids(tokenizer, cand)
            if ids:
                t = ids[-1]
                if t not in special_ids:
                    filler_id = t
                    break
        if filler_id is None and nospace_ids:
            filler_id = nospace_ids[0]

    def try_add_space_like(chs: List[str]):
        for ch in chs:
            ids = encode_ids(tokenizer, ch)
            if not ids:
                continue
            t = ids[-1]
            dec = decode_ids(tokenizer, [t])
            if is_allowed_text(dec):
                space_ids.append(t)
                if len(space_ids) >= 3:
                    return

    def try_add_nospace_like(chs: List[str]):
        for ch in chs:
            ids = encode_ids(tokenizer, ch)
            if not ids:
                continue
            t = ids[-1]
            dec = decode_ids(tokenizer, [t])
            if is_allowed_text(dec) and not dec.startswith(" "):
                nospace_ids.append(t)
                if len(nospace_ids) >= 2:
                    return

    if len(space_ids) < 3:
        try_add_space_like([" 0", " 1", " a", " b", " A", " B", " 2", " 3", " X", " Z"])
    if len(nospace_ids) < 2:
        try_add_nospace_like(["a", "b", "A", "B", "Z", "X", "0", "1", "2"])

    if not space_ids and filler_id is not None:
        space_ids.append(filler_id)
    if not nospace_ids and filler_id is not None:
        nospace_ids.append(filler_id)

    if filler_id is None:
        for tid in range(vocab_size):
            if tid in special_ids:
                continue
            filler_id = tid
            break

    return nospace_ids, space_ids, (filler_id if filler_id is not None else 0)

def fix_to_target_token_len_by_ids(tokenizer, ids: List[int], target_len: int,
                                   add_token_id: int) -> List[int]:
    """
    单次 decode→过滤→re-encode 后，直接在 ID 层面 pad/truncate 到 target_len。
    避免反复调用 tokenizer，将 4096 次循环降为 1~2 次 tokenizer 调用。
    """
    if not ids:
        ids = [add_token_id]

    # 单次闭环：decode → filter → encode
    text = decode_ids(tokenizer, ids)
    text = filter_allowed(text)
    cur_ids = encode_ids(tokenizer, text)

    # 直接在 ID 层面修正长度，不再反复 decode/encode
    if len(cur_ids) < target_len:
        cur_ids.extend([add_token_id] * (target_len - len(cur_ids)))
    elif len(cur_ids) > target_len:
        cur_ids = cur_ids[:target_len]

    # 最终一致性检查：确保过滤后的文本再 encode 长度不变
    final_text = filter_allowed(decode_ids(tokenizer, cur_ids))
    final_ids = encode_ids(tokenizer, final_text)

    # 如仍有微小偏差（通常 <3），在 ID 层面快速修正
    if len(final_ids) < target_len:
        final_ids.extend([add_token_id] * (target_len - len(final_ids)))
    elif len(final_ids) > target_len:
        final_ids = final_ids[:target_len]

    return final_ids

def make_prefix_ids(tokenizer, nospace_ids: List[int], space_ids: List[int],
                    base_len: int, rng: random.Random, filler_id: int) -> List[int]:
    """
    生成长度为 base_len 的"公共前缀" ids：首 token 选 nospace，其余优先选 space。
    """
    ids = []
    first = (nospace_ids[0] if nospace_ids else filler_id)
    ids.append(first)
    for _ in range(max(0, base_len - 1)):
        ids.append(space_ids[rng.randrange(len(space_ids))] if space_ids else filler_id)
    ids = fix_to_target_token_len_by_ids(tokenizer, ids, base_len,
                                         add_token_id=(space_ids[0] if space_ids else filler_id))
    return ids

def idx_to_bit_ids(idx: int, bits: int, bit0_id: int, bit1_id: int) -> List[int]:
    if bits <= 0:
        return []
    s = format(idx, f"0{bits}b")
    return [bit0_id if ch == "0" else bit1_id for ch in s]

def parse_prefix_ratio(r: str) -> float:
    """
    "50%" -> 0.5, "0.5" -> 0.5, "0.500" -> 0.5
    """
    r = str(r).strip()
    if r.endswith("%"):
        v = float(r[:-1]) / 100.0
    else:
        v = float(r)
    if not (0.0 <= v <= 1.0):
        raise ValueError("prefix-ratio 必须在 [0,1] 区间或百分数 [0%,100%]")
    return v


# ------------------------- 长度分布（变长模式） -------------------------

def sample_target_length(
    rng: random.Random,
    fixed_length: int,
    length_mean: Optional[int] = None,
    length_std: Optional[float] = None,
    length_min: Optional[int] = None,
    length_max: Optional[int] = None,
) -> int:
    fixed_length = max(1, int(fixed_length))
    has_gauss = (length_mean is not None) and (length_std is not None)
    has_range = (length_min is not None) and (length_max is not None)

    lo = 1 if length_min is None else max(1, int(length_min))
    hi = None if length_max is None else max(1, int(length_max))
    if hi is not None and lo > hi:
        lo, hi = hi, lo

    if has_gauss:
        mu = max(1, int(length_mean))
        sigma = max(0.0, float(length_std))
        val = mu if sigma == 0 else int(round(rng.gauss(mu, sigma)))
        if hi is not None:
            val = min(val, hi)
        val = max(lo, val)
        return max(1, val)

    if has_range:
        return rng.randint(lo, hi)

    return fixed_length


def build_length_tag(
    fixed_length: int,
    length_mean: Optional[int],
    length_std: Optional[float],
    length_min: Optional[int],
    length_max: Optional[int],
) -> str:
    if (length_mean is not None) and (length_std is not None):
        tag = f"G{int(length_mean)}_{str(length_std).replace('.', 'd')}"
        if (length_min is not None) and (length_max is not None):
            tag += f"_C{int(length_min)}_{int(length_max)}"
        return tag
    if (length_min is not None) and (length_max is not None):
        return f"U{int(length_min)}_{int(length_max)}"
    return f"L{int(fixed_length)}"


# ------------------------- 主流程 -------------------------

def create_multi_prefix_dataset(
    data_num, prefix_num, length, ratio, model_path, seeds, dataset_path, dp_size,
    length_mean: Optional[int] = None,
    length_std: Optional[float] = None,
    length_min: Optional[int] = None,
    length_max: Optional[int] = None,
):
    total = max(1, int(data_num))
    num_prefixes = max(1, int(prefix_num))
    tokens = max(1, int(length))
    hit_ratio = parse_prefix_ratio(ratio)
    seed = int(seeds)
    rng = random.Random(seed)

    # 判断是否启用变长模式
    use_variable_length = (
        (length_mean is not None and length_std is not None)
        or (length_min is not None and length_max is not None)
    )

    # 加载 tokenizer（fast 优先）
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    nospace_ids, space_ids, filler_id = build_safe_token_pools(tokenizer)
    append_id = space_ids[0] if space_ids else filler_id

    # -------------------- 变长模式 --------------------
    if use_variable_length:
        # 预采样所有 real_len
        real_lens: List[int] = []
        for _ in range(total):
            rl = sample_target_length(
                rng=rng,
                fixed_length=tokens,
                length_mean=length_mean,
                length_std=length_std,
                length_min=length_min,
                length_max=length_max,
            )
            real_lens.append(max(1, int(rl)))

        common_lens = [max(0, min(rl, int(round(rl * hit_ratio)))) for rl in real_lens]
        max_common_len = max(common_lens) if common_lens else 0

        # 生成公共前缀池（统一长度 max_common_len）
        first_id = nospace_ids[0] if nospace_ids else filler_id
        prefix_pool_ids: List[List[int]] = []
        for _ in range(num_prefixes):
            if max_common_len <= 0:
                pids = []
            else:
                pids = [first_id]
                for _j in range(max_common_len - 1):
                    tid = space_ids[rng.randrange(len(space_ids))] if space_ids else filler_id
                    pids.append(tid)
                pids = fix_to_target_token_len_by_ids(tokenizer, pids, max_common_len, add_token_id=append_id)
            prefix_pool_ids.append(pids)

        # 命名
        tok_name = basename_of(model_path)
        pct_label = f"{int(round(hit_ratio * 100))}p"
        length_tag = build_length_tag(tokens, length_mean, length_std, length_min, length_max)
        base_tag = f"seed{seed}_{length_tag}_{pct_label}_num{total}_pn{num_prefixes}_dp{dp_size}_{tok_name}"

        ensure_dir(dataset_path)
        prefix_dir = os.path.join(dataset_path, f"{base_tag}_prefix")
        ensure_dir(prefix_dir)

        prefix_jsonl_path = os.path.join(prefix_dir, "prefix.jsonl")
        dataset_jsonl_path = os.path.join(dataset_path, f"{base_tag}.jsonl")

        # 缓存检查
        if os.path.exists(prefix_jsonl_path) and os.path.exists(dataset_jsonl_path):
            try:
                with open(prefix_jsonl_path, "r", encoding="utf-8") as pf:
                    prefix_lines = sum(1 for _ in pf if _.strip())
                with open(dataset_jsonl_path, "r", encoding="utf-8") as df:
                    dataset_lines = sum(1 for _ in df if _.strip())
                if prefix_lines >= num_prefixes * dp_size and dataset_lines >= total:
                    print(f"[缓存命中] 数据集已存在，跳过生成：")
                    print(f"  - 公共前缀：{prefix_jsonl_path}")
                    print(f"  - 数据集：  {dataset_jsonl_path}")
                    return {
                        "prefix_jsonl": prefix_jsonl_path,
                        "dataset_jsonl": dataset_jsonl_path,
                        "max_common_len": max_common_len,
                        "avg_hit_ratio": (sum((c / r) for c, r in zip(common_lens, real_lens)) / len(real_lens)) if real_lens else 0.0,
                    }
            except Exception:
                pass

        # 写 prefix.jsonl（每个前缀重复 dp_size 次）
        with open(prefix_jsonl_path, "w", encoding="utf-8") as pf:
            for pids in prefix_pool_ids:
                ptxt = decode_ids(tokenizer, pids) if pids else ""
                if not is_allowed_text(ptxt):
                    ptxt = filter_allowed(ptxt)
                for _ in range(dp_size):
                    pf.write(json.dumps({"question": ptxt, "answer": ""}, ensure_ascii=True))
                    pf.write("\n")

        # 生成 dataset.jsonl
        remaining = total
        groups_left = num_prefixes
        sample_idx = 0

        with open(dataset_jsonl_path, "w", encoding="utf-8") as df:
            for g_idx in range(num_prefixes):
                if remaining <= 0 or sample_idx >= total:
                    break

                group_target = int(math.ceil(remaining / groups_left))
                groups_left -= 1
                base_prefix_ids = prefix_pool_ids[g_idx] if g_idx < len(prefix_pool_ids) else []

                for _ in range(group_target):
                    if sample_idx >= total:
                        break

                    real_len = real_lens[sample_idx]
                    common_len = common_lens[sample_idx]

                    # 前缀截取
                    prefix_part = base_prefix_ids[:common_len] if common_len > 0 else []
                    cur_ids = list(prefix_part)

                    # 补后缀
                    suffix_need = max(0, real_len - len(cur_ids))
                    for _k in range(suffix_need):
                        tid = space_ids[rng.randrange(len(space_ids))] if space_ids else filler_id
                        cur_ids.append(tid)

                    # 严格长度对齐
                    final_ids = fix_to_target_token_len_by_ids(
                        tokenizer, cur_ids, real_len, add_token_id=append_id
                    )

                    q = decode_ids(tokenizer, final_ids)
                    if not is_allowed_text(q):
                        q = filter_allowed(q)
                        q_ids = encode_ids(tokenizer, q)
                        q_ids = fix_to_target_token_len_by_ids(tokenizer, q_ids, real_len, add_token_id=append_id)
                        q = decode_ids(tokenizer, q_ids)

                    df.write(json.dumps({"question": q, "answer": ""}, ensure_ascii=True))
                    df.write("\n")
                    sample_idx += 1

                remaining -= group_target

        return {
            "prefix_jsonl": prefix_jsonl_path,
            "dataset_jsonl": dataset_jsonl_path,
            "max_common_len": max_common_len,
            "avg_hit_ratio": (sum((c / r) for c, r in zip(common_lens, real_lens)) / len(real_lens)) if real_lens else 0.0,
        }

    # -------------------- 定长模式（pro 原有逻辑，保持不变） --------------------
    # 计算前缀长度（四舍五入到最邻近整数，且夹紧到 [0, tokens]）
    prefix_len = int(round(tokens * hit_ratio))
    if prefix_len < 0:
        prefix_len = 0
    if prefix_len > tokens:
        prefix_len = tokens

    # 选择 bit0/bit1：优先两个不同的"空格型安全 token"；不足再用 nospace/filler
    bit0_id = (space_ids[0] if len(space_ids) >= 1 else filler_id)
    bit1_id = (space_ids[1] if len(space_ids) >= 2 else (nospace_ids[0] if nospace_ids else filler_id))
    if bit1_id == bit0_id:
        pool = (space_ids + nospace_ids) or [filler_id]
        for t in pool:
            if t != bit0_id:
                bit1_id = t
                break

    # 每条样本允许的"位段"长度（确保"额外公共前缀"<=16）
    # 若剩余空间不足，则 bits 下降；若为 0 则跳过位段
    max_extra = 16
    bits = min(max_extra, max(0, tokens - prefix_len))

    tok_name = basename_of(model_path)
    pct_label = f"{int(round(hit_ratio * 100))}p"  # 命名用 50p 表示 50%
    # 文件名包含 prefix_num 和 dp_size，避免参数变化时缓存冲突
    base_tag = f"seed{seed}_in{tokens}_{pct_label}_num{total}_pn{num_prefixes}_dp{dp_size}_{tok_name}"

    # 输出路径
    prefix_dir = os.path.join(dataset_path, f"{base_tag}_prefix")
    ensure_dir(prefix_dir)
    prefix_jsonl_path = os.path.join(prefix_dir, "prefix.jsonl")
    dataset_jsonl_path = os.path.join(dataset_path, f"{base_tag}.jsonl")

    # 缓存检查：如果文件已存在且行数足够，直接返回
    if os.path.exists(prefix_jsonl_path) and os.path.exists(dataset_jsonl_path):
        try:
            with open(prefix_jsonl_path, "r", encoding="utf-8") as pf:
                prefix_lines = sum(1 for _ in pf if _.strip())
            with open(dataset_jsonl_path, "r", encoding="utf-8") as df:
                dataset_lines = sum(1 for _ in df if _.strip())
            if prefix_lines >= num_prefixes and dataset_lines >= total:
                print(f"[缓存命中] 数据集已存在，跳过生成：")
                print(f"  - 公共前缀：{prefix_jsonl_path}")
                print(f"  - 数据集：  {dataset_jsonl_path}")
                return prefix_jsonl_path, dataset_jsonl_path
        except Exception:
            pass  # 校验失败则重新生成

    # 生成 num_prefixes 个"公共前缀"
    prefix_list_ids: List[List[int]] = []
    for _ in range(num_prefixes):
        if prefix_len == 0:
            prefix_ids = []
        else:
            prefix_ids = make_prefix_ids(tokenizer, nospace_ids, space_ids, prefix_len, rng, filler_id)
        prefix_list_ids.append(prefix_ids)

    # 直接用内存里的 prefix 文本（make_prefix_ids 已经 fix 过了）
    loaded_prefix_texts: List[str] = []
    for pids in prefix_list_ids:
        ptxt = decode_ids(tokenizer, pids) if pids else ""
        if not is_allowed_text(ptxt):
            ptxt = filter_allowed(ptxt)
        loaded_prefix_texts.append(ptxt)

    # 写公共前缀文件（GSM8K格式）
    try:
        with open(prefix_jsonl_path, "w", encoding="utf-8") as pf:
            for ptxt in loaded_prefix_texts:
                for _ in range(dp_size):
                    pf.write(json.dumps({"question": ptxt, "answer": ""}, ensure_ascii=True))
                    pf.write("\n")
    except Exception:
        # 兜底：纯 A0 A1 ... 文本填充
        with open(prefix_jsonl_path, "w", encoding="utf-8") as pf:
            for _ in range(num_prefixes):
                seed_text = " ".join(f"A{i%10}" for i in range(max(1, prefix_len))) if prefix_len > 0 else ""
                fixed_ids = fix_to_target_token_len_by_ids(
                    tokenizer, encode_ids(tokenizer, seed_text), prefix_len,
                    add_token_id=(space_ids[0] if space_ids else filler_id)
                ) if prefix_len > 0 else []
                ptxt = decode_ids(tokenizer, fixed_ids) if fixed_ids else ""
                pf.write(json.dumps({"question": ptxt, "answer": ""}, ensure_ascii=True))
                pf.write("\n")

    # 生成数据集（总计 total 行），按"逐组 ceil 配额，最后一组可能更少"的策略分配
    remaining = total
    groups_left = num_prefixes

    progress = tqdm(total=total, desc="Generating dataset", unit="row") if tqdm else None
    try:
        with open(dataset_jsonl_path, "w", encoding="utf-8") as df:
            for g_idx in range(num_prefixes):
                if remaining <= 0:
                    break
                group_target = int(math.ceil(remaining / groups_left))  # 该组计划条数
                groups_left -= 1
                # 该组使用的前缀文本
                prefix_text = loaded_prefix_texts[g_idx] if g_idx < len(loaded_prefix_texts) else ""
                prefix_ids_fixed = encode_ids(tokenizer, prefix_text) if prefix_text else []

                for i in range(group_target):
                    # 位段（控制"额外公共前缀"<=16）
                    bit_ids = idx_to_bit_ids(i, bits=bits, bit0_id=bit0_id, bit1_id=bit1_id)
                    cur_ids = list(prefix_ids_fixed) + bit_ids

                    # 先用随机"空格型安全 token"填充至接近目标，再闭环修正
                    # 直接用 len(cur_ids) 估算，避免多余的 decode→filter→encode
                    remain_len = max(0, tokens - len(cur_ids))
                    for _ in range(remain_len):
                        tid = (space_ids[rng.randrange(len(space_ids))] if space_ids else filler_id)
                        cur_ids.append(tid)

                    final_ids = fix_to_target_token_len_by_ids(
                        tokenizer, cur_ids, tokens,
                        add_token_id=(space_ids[0] if space_ids else filler_id)
                    )
                    q = decode_ids(tokenizer, final_ids)
                    if not is_allowed_text(q):
                        q = filter_allowed(q)
                    df.write(json.dumps({"question": q, "answer": ""}, ensure_ascii=True))
                    df.write("\n")
                    if progress:
                        progress.update(1)

                remaining -= group_target
    finally:
        if progress:
            progress.close()
    return prefix_jsonl_path, dataset_jsonl_path
