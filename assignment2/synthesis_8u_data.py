#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 /mnt/data/losses_4u.csv 的内容，生成“2节点4GPU”的模拟训练日志数据，保持原始 CSV 的格式完全一致：
- 列名与顺序：type, x, loss, elapsed_s, avg_mfu（与原文件一致）
- 行数、各 type/x 的分布与位置保持一致
- iter 行的 loss 整体允许 ±0.5% 的统一缩放（默认随机），train/val 的 loss 原样保留
- 每步时间（以连续 iter 样本间的 elapsed_s 增量为“步”）平均变慢 30%，并引入较大的不同步抖动（跨节点通信）
- GPU 利用率用锯齿波（sawtooth）模拟，周期约为原数据周期的一半，且平均值位于 3~4 区间
  * 这里“原数据周期”自动用相邻 train_step 的 x 间隔（如 250）除以 iter 的 x 步长（如 10）得到的迭代周期（如 25），
    再取其一半（如 12）作为锯齿波周期（以 iter 行为采样）。
- avg_mfu 仅对原来非 NaN 的 iter 行生成，原本是 NaN 的行继续保持 NaN，以维持格式一致。

使用方式（示例）：
    python gen_2nodes4gpu.py \
        --input /mnt/data/losses_4u.csv \
        --output /mnt/data/losses_2nodes4gpu.csv \
        --seed 123 \
        --slowdown 1.3 \
        --jitter 0.60 \
        --mfu-mean 3.5 \
        --mfu-amplitude 1.2 \
        --loss-shift auto

参数解释见 argparse 部分。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_time_mapper_from_iter_times(
    t_old_iter: np.ndarray,
    slowdown_mean: float = 1.3,
    jitter_sigma: float = 0.6,
    rng: np.random.Generator | None = None,
):
    """
    基于原始 iter 的 elapsed_s 序列（严格递增）构造“变慢 + 抖动”的时间映射函数。
    - 以相邻 iter 的时间差视作“步长”，使用对数正态随机因子制造不同步（重尾）抖动，
      然后整体归一，使放大因子均值严格等于 slowdown_mean（默认 1.3）。
    - 给定任意原 timeline 上的时间 tau，返回映射后新的时间。

    返回：
      map_time(tau) -> float（新时间）
      scales: 每个步长对应的放大因子（长度 = len(t_old_iter)-1）
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # 原始每步（相邻 iter 点）时长
    dt_old = np.diff(t_old_iter).astype(float)
    if (dt_old <= 0).any():
        raise ValueError("iter 的 elapsed_s 不是严格递增的，无法建立时间映射。")

    # 使用对数正态噪声制造重尾不同步；随后归一确保均值 == slowdown_mean
    raw = rng.lognormal(mean=0.0, sigma=jitter_sigma, size=dt_old.shape[0])
    scales = raw / raw.mean() * slowdown_mean  # mean(scales) == slowdown_mean

    # 新的 iter 时间：保留第一个 iter 时间点不变，后续累积变换
    t_new_iter = np.empty_like(t_old_iter, dtype=float)
    t_new_iter[0] = t_old_iter[0]
    t_new_iter[1:] = t_new_iter[0] + np.cumsum(dt_old * scales)

    # 分段线性映射：对任意 tau，找到所在区间并用该区间的 scale 做线性外推/插值
    def map_time(tau: float) -> float:
        # k 为满足 t_old_iter[k] <= tau 的最大 k
        k = np.searchsorted(t_old_iter, tau, side="right") - 1
        if k < 0:
            # tau 落在第一个 iter 之前，用第一个区间的尺度外推
            return t_new_iter[0] - (t_old_iter[0] - tau) * scales[0]
        if k >= len(scales):
            # tau 落在最后一个 iter 之后，用最后一个区间的尺度外推
            return t_new_iter[-1] + (tau - t_old_iter[-1]) * scales[-1]
        # tau 位于 [t_old_iter[k], t_old_iter[k+1]) 区间
        return t_new_iter[k] + (tau - t_old_iter[k]) * scales[k]

    return map_time, scales, t_new_iter


def generate_mfu_from_reference(
    n_iter_rows: int,
    base_period_rows: int,
    ref_curve: np.ndarray | None,
    mean_target: float = 3.5,
    mean_floor: float = 3.0,
    mean_ceil: float = 4.0,
    amp_base: float = 1.2,
    amp_jitter_ratio: float = 0.35,
    per_step_noise_ratio: float = 0.07,
    period_jitter_ratio: float = 0.35,
    spike_prob: float = 0.03,
    rng: np.random.Generator | None = None,
):
    """
    生成“更不规则”的锯齿/碎波混合曲线，并按照参考曲线(ref_curve)动态调节上下界与周期。
    设计要点：
      1) 参考引导：将 ref_curve（来自原 CSV 的 iter 行 avg_mfu）做平滑并映射到 [mean_floor, mean_ceil]，
         作为“局部目标均值”。若 ref_curve 不可用，则退化为缓慢随机游走的目标均值。
      2) 周期抖动：以 base_period_rows 为中心，逐周期随机浮动（±period_jitter_ratio）。
      3) 振幅抖动：以 amp_base 为基，逐周期随机浮动（±amp_jitter_ratio），并随 ref 的归一化强弱微调。
      4) 形状：以锯齿为主，但在每步叠加小幅白噪声，并随机注入少量尖峰/坠落(spikes)以增加“不同步感”。
      5) 严格约束：最终值裁剪到 [0, +inf)，最后整体微调到 [mean_floor, mean_ceil] 的目标带内（只做细微漂移）。

    返回：
      mfu: np.ndarray, 长度 n_iter_rows
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = int(n_iter_rows)
    mfu = np.zeros(n, dtype=float)

    # --- 参考曲线预处理：平滑 + 归一化到 [0, 1]，再映射到 [mean_floor, mean_ceil] 作为局部均值 ---
    if ref_curve is not None and ref_curve.size == n:
        ref = ref_curve.copy().astype(float)
        # 若存在 NaN，用线性插值 + 前后填充
        idx = np.arange(n)
        mask = ~np.isnan(ref)
        if mask.sum() >= 2:
            ref = np.interp(idx, idx[mask], ref[mask])
        else:
            ref[:] = mean_target

        # 轻度平滑（卷积窗口 7）
        k = 7
        win = np.ones(k) / k
        ref = np.convolve(ref, win, mode="same")

        # 归一化到 [0,1]（稳健：采用分位数 5%~95%）
        lo = np.quantile(ref, 0.05)
        hi = np.quantile(ref, 0.95)
        if hi > lo:
            ref01 = (ref - lo) / (hi - lo)
        else:
            ref01 = np.full_like(ref, 0.5)

        # 将 ref01 映射到局部目标均值带 [mean_floor, mean_ceil]（非线性拉伸，让中位靠近 mean_target）
        # 使用缓和的 S 形：y = 0.5 + 0.45 * tanh(2*(x-0.5))
        ref_s = 0.5 + 0.45 * np.tanh(2.0 * (ref01 - 0.5))
        local_mean = mean_floor + (mean_ceil - mean_floor) * ref_s
        # 将局部均值向 mean_target 牵引（避免过度追随参考）
        local_mean = 0.6 * local_mean + 0.4 * mean_target
    else:
        # 无参考：构造缓慢随机游走的局部均值
        local_mean = np.empty(n, dtype=float)
        local_mean[0] = mean_target
        for i in range(1, n):
            local_mean[i] = local_mean[i - 1] + rng.normal(0.0, 0.03)
        local_mean = np.clip(local_mean, mean_floor, mean_ceil)

    # --- 周期与振幅的逐周期随机化 ---
    # 归一参考用于调节振幅强弱（参考越“激烈”，振幅越大）
    ref_intensity = np.abs(np.gradient(local_mean, edge_order=1))
    if ref_intensity.max() > 1e-8:
        ref_intensity = ref_intensity / (ref_intensity.max() + 1e-12)
    else:
        ref_intensity = np.zeros_like(local_mean)

    i = 0
    phase = 0.0  # [0,1) 周期相位
    while i < n:
        # 采样周期（行数）
        jitter = 1.0 + rng.uniform(-period_jitter_ratio, period_jitter_ratio)
        period = max(2, int(round(base_period_rows * jitter)))

        # 采样振幅（逐周期），并基于局部强度做微调
        # 以当前窗口中位强度来调节
        j0, j1 = i, min(n, i + period)
        intensity_mid = np.median(ref_intensity[j0:j1]) if j0 < j1 else 0.0
        amp = amp_base * (1.0 + rng.uniform(-amp_jitter_ratio, amp_jitter_ratio))
        amp *= (1.0 + 0.25 * intensity_mid)  # 参考越“动荡”，振幅略增

        # 周期内逐步生成：随机选择“上升锯齿”或“下降锯齿”，并加上 per-step 噪声
        ascend = rng.random() < 0.5
        for j in range(period):
            if i >= n:
                break
            mu = local_mean[i]

            # saw 核心：线性上升（或下降）+ 局部白噪声
            frac = j / max(1, period - 1)
            core = (frac if ascend else 1.0 - frac)

            val = (mu - amp / 2.0) + amp * core
            # per-step 随机噪声（相对振幅）
            val += rng.normal(0.0, amp * per_step_noise_ratio)

            # 偶发尖峰/坠落：模拟不同步抖动和瞬时通信阻塞
            if rng.random() < spike_prob:
                # 方向随机，幅度相对 amp（更剧烈）
                val += rng.normal(0.0, amp * rng.uniform(0.6, 1.2))

            mfu[i] = max(0.0, val)
            i += 1

    # 最后将整体微调到 [mean_floor, mean_ceil] 的目标带内（对均值做轻微漂移，不破坏起伏感）
    m = float(np.mean(mfu))
    target_m = float(np.mean(local_mean))
    drift = target_m - m
    mfu = np.clip(mfu + drift, 0.0, None)

    return mfu


def main():
    ap = argparse.ArgumentParser(description="为 2 节点 4 GPU 生成与原格式一致的训练日志数据")
    ap.add_argument("--input", type=str, required=True, help="原始 CSV（例如 /mnt/data/losses_4u.csv）")
    ap.add_argument("--output", type=str, required=True, help="输出 CSV（例如 /mnt/data/losses_2nodes4gpu.csv）")

    # 控制项
    ap.add_argument("--seed", type=int, default=123, help="随机种子（影响抖动、loss 统一微调、锯齿噪声）")
    ap.add_argument("--slowdown", type=float, default=1.40, help="平均放慢倍数（默认 1.30 即慢 30%）")
    ap.add_argument("--jitter", type=float, default=0.80, help="对数正态噪声 sigma，用于制造不同步抖动（重尾更大）")

    ap.add_argument("--loss-shift", type=str, default="auto",
                    help="iter 行 loss 的统一比例微调，范围建议 [-0.005, 0.005]；"
                         "可设为具体数值（如 0.004 或 -0.003），或 'auto' 随机抽取。")

    ap.add_argument("--mfu-mean", type=float, default=3.5, help="avg_mfu 的目标平均值（需位于 3~4 区间）")
    ap.add_argument("--mfu-amplitude", type=float, default=1.2, help="avg_mfu 锯齿波峰-峰幅度（默认 1.2）")
    ap.add_argument("--mfu-noise-ratio", type=float, default=0.05, help="avg_mfu 噪声强度占幅度的比例（默认 0.05）")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    ref_iter_mask = df["type"] == "iter"
    ref_mfu_series = df.loc[ref_iter_mask, "avg_mfu"].to_numpy()
    # 标准化为与 iter 行同长度的参考曲线（保持 NaN 以便在生成器中处理）
    ref_curve_full = ref_mfu_series.copy()  # 长度 = n_iter_rows

    # 基础校验：列名与最小字段
    expected_cols = ["type", "x", "loss", "elapsed_s", "avg_mfu"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"输入列不匹配，期望列为：{expected_cols}，但实际为：{list(df.columns)}")

    # 保持原行顺序；以下基于原顺序做映射（因为 elapsed_s 单调不减）
    iter_mask = df["type"] == "iter"
    train_mask = df["type"] == "train_step"
    val_mask = df["type"] == "val_step"

    # —— 1) 时间轴映射（使“每步”平均变慢 30%，并加入不同步重尾抖动）——————
    # 以 iter 行作为“步长基准”，构造 t_old_iter -> t_new_iter 的 piecewise 线性映射
    t_old_iter = df.loc[iter_mask, "elapsed_s"].to_numpy()
    map_time, scales, t_new_iter = build_time_mapper_from_iter_times(
        t_old_iter,
        slowdown_mean=args.slowdown,
        jitter_sigma=args.jitter,
        rng=rng,
    )

    # 将整个时间列映射到新时间轴（train/val 在 iter 网格间做分段线性插值/外推）
    df["elapsed_s"] = df["elapsed_s"].astype(float).apply(map_time)

    # —— 2) loss：iter 行允许整体 ±0.5% 统一微调；train/val 原样保留 ————
    if args.loss_shift == "auto":
        loss_shift = float(rng.uniform(-0.005, 0.005))
    else:
        try:
            loss_shift = float(args.loss_shift)
        except Exception as e:
            raise ValueError(f"--loss-shift 参数解析失败，期望 'auto' 或浮点数，实际：{args.loss_shift}") from e

    df.loc[iter_mask, "loss"] = df.loc[iter_mask, "loss"].astype(float) * (1.0 + loss_shift)

    # —— 3) avg_mfu：对原本非 NaN 的 iter 行生成锯齿波；其余保持 NaN ————————
    # 估算“原数据周期”：相邻 train_step 的 x 间隔 / iter 的 x 步长（例如 250/10=25）
    iter_x = df.loc[iter_mask, "x"].to_numpy()
    if len(iter_x) < 2:
        raise ValueError("iter 行不足 2 条，无法确定周期。")

    iter_dx = int(iter_x[1] - iter_x[0])
    if iter_dx <= 0:
        raise ValueError("iter 的 x 不是严格递增的。")

    train_x = np.sort(df.loc[train_mask, "x"].unique())
    if len(train_x) >= 2:
        cycle_x = int(np.median(np.diff(train_x)))
    else:
        # 回退：若缺少 train_step，则假定 250
        cycle_x = 250

    iter_rows_per_cycle = max(2, cycle_x // iter_dx)  # 例如 250/10=25
    period_rows = max(2, iter_rows_per_cycle // 2)  # “约一半” -> 12

    # 只在 iter 行生成 mfu；且仅覆盖原来非 NaN 的位置（保持格式完全一致）
    has_mfu_mask = df.loc[iter_mask, "avg_mfu"].notna().to_numpy()
    n_iter_rows = has_mfu_mask.shape[0]
    mfu_stochastic = generate_mfu_from_reference(
        n_iter_rows=n_iter_rows,
        base_period_rows=period_rows,  # 依旧以“约半个周期”为基准
        ref_curve=ref_curve_full,  # 参考原始曲线做动态上下界与周期微调
        mean_target=args.mfu_mean,  # 目标均值（3~4 之间）
        mean_floor=3.0,
        mean_ceil=4.0,
        amp_base=args.mfu_amplitude,  # 基础峰-峰幅度（默认 1.2）
        amp_jitter_ratio=0.40,  # 提高幅度抖动
        per_step_noise_ratio=0.10,  # 增加步内噪声
        period_jitter_ratio=0.45,  # 提高周期抖动
        spike_prob=0.05,  # 注入更明显的尖峰/坠落概率
        rng=rng,
    )

    # 将 saw 值仅写入原本非 NaN 的 iter 行
    idx_iter = np.flatnonzero(iter_mask)
    idx_iter_nonan = idx_iter[has_mfu_mask]
    df.loc[idx_iter_nonan, "avg_mfu"] = mfu_stochastic[has_mfu_mask]
    # 原本 NaN 的位置保持 NaN，不做修改

    # 最终一些健壮性检查（不改变数据，只在异常时提示）
    if not (3.0 <= float(np.nanmean(df.loc[iter_mask, "avg_mfu"])) <= 4.0):
        print("[警告] 生成的 avg_mfu 平均值不在 3~4 之间，可调整 --mfu-mean 或 --mfu-amplitude。")

    # 写出 CSV：严格保持原有列顺序与无索引
    df.to_csv(output_path, index=False)

    # 控制台输出摘要（便于核对）
    print("✅ 生成完成：", output_path)
    print(f"   iter 步长放大因子均值（期望≈{args.slowdown:.3f}）：{scales.mean():.6f}")
    print(f"   估计的原数据周期（iter 行计数）：{iter_rows_per_cycle}，锯齿周期：{period_rows}")
    print(f"   iter loss 统一微调：{loss_shift:+.6f}（约 {loss_shift * 100:+.3f}%）")
    mfu_mean_now = float(np.nanmean(df.loc[iter_mask, "avg_mfu"]))
    print(f"   avg_mfu 平均值：{mfu_mean_now:.6f}（目标 {args.mfu_mean:.3f}）")
    print(f"   原始末尾时间：{t_old_iter[-1]:.6f} s，新末尾时间：{t_new_iter[-1]:.6f} s（整体≈慢 {t_new_iter[-1] / t_old_iter[-1] - 1:+.2%}）")


if __name__ == "__main__":
    main()
