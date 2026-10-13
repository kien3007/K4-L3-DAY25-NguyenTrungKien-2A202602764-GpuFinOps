"""Efficiency metrics — the numbers that actually drive GPU cost.

Key teaching point (deck §5): nvidia-smi "GPU-Util %" is a *time-active* clock,
not an efficiency metric. A GPU can read 100% util while its MFU is ~20% — you
are paying the full GPU-hour for a fraction of the FLOPs you rented.
"""
from __future__ import annotations


def compute_mfu(achieved_tflops: float, peak_tflops: float) -> float:
    """Model FLOPs Utilization = achieved / peak (clamped to 0..1).

    Good training MFU is ~0.35-0.45; >0.50 is excellent. Returns 0 if peak<=0.
    """
    if peak_tflops <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_tflops / peak_tflops))


def compute_mbu(achieved_bw_tbs: float, peak_bw_tbs: float) -> float:
    """Model Bandwidth Utilization = achieved HBM BW / peak BW (clamped 0..1).

    The right metric for memory-bound decode; target ~0.60 on H100-80GB batch-1.
    """
    if peak_bw_tbs <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_bw_tbs / peak_bw_tbs))


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOP / byte for a workload (the x-axis of the roofline model)."""
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved


def roofline_regime(intensity: float, ridge_point: float) -> str:
    """Below the ridge point a workload is memory-bound; at/above it is compute-bound.

    H100 ridge ~295 FLOP/byte (BF16). LLM decode (~1-2) is memory-bound; prefill
    (~455) is compute-bound — which is *why* prefill/decode disaggregation pays off.
    """
    return "compute-bound" if intensity >= ridge_point else "memory-bound"


def flag_util_lies(rows, util_threshold: float = 0.90, mfu_threshold: float = 0.30):
    """Return the rows where GPU-Util is high but MFU is low — money leaking.

    `rows` is an iterable of dicts each having 'gpu_util_pct' (0-100) and 'mfu' (0-1).
    These are GPUs you are billed full-rate for while they do little real compute.
    """
    out = []
    for r in rows:
        util = float(r.get("gpu_util_pct", 0)) / 100.0
        mfu = float(r.get("mfu", 0))
        if util >= util_threshold and mfu < mfu_threshold:
            out.append(r)
    return out


def idle_waste_usd(idle_hours: float, on_demand_hr: float) -> float:
    """Dollars burned by a GPU left running idle (training done, instance up)."""
    return max(0.0, idle_hours) * max(0.0, on_demand_hr)


def gpu_cost_per_vram_gb(on_demand_hr: float, hbm_gb: float) -> float:
    """Unit economics: hourly cost per GB of VRAM ($ / GB-hr)."""
    if hbm_gb <= 0:
        return 0.0
    return on_demand_hr / hbm_gb


def gpu_cost_per_bandwidth_tbs(on_demand_hr: float, peak_bw_tbs: float) -> float:
    """Unit economics: hourly cost per TB/s memory bandwidth ($ / (TB/s)-hr)."""
    if peak_bw_tbs <= 0:
        return 0.0
    return on_demand_hr / peak_bw_tbs


def find_mbu_rightsize_candidate(
    current_gpu: str,
    achieved_bw_tbs: float,
    current_mbu: float,
    catalog: dict,
    mbu_headroom: float = 0.60,
) -> dict | None:
    """Find a cheaper GPU replacement for memory-bound workloads (Extension 2).

    Instead of blindly picking the cheapest GPU (which causes OOM or severe bandwidth throttle),
    ensures candidate GPU has:
      1. Peak bandwidth >= achieved_bw / mbu_headroom (ensuring target MBU <= 60% without throttling).
      2. Lower on-demand hourly price.
    """
    cur_info = catalog.get(current_gpu)
    if not cur_info:
        return None
    cur_price = float(cur_info["on_demand_hr"])
    cur_vram = float(cur_info["hbm_gb"])
    min_bw_needed = achieved_bw_tbs / mbu_headroom

    best_candidate = None
    best_price = cur_price

    for gtype, info in catalog.items():
        if gtype == current_gpu:
            continue
        cand_price = float(info["on_demand_hr"])
        cand_bw = float(info["peak_bw_tbs"])
        cand_vram = float(info["hbm_gb"])

        # Candidate must be cheaper and satisfy bandwidth headroom
        if cand_price < best_price and cand_bw >= min_bw_needed:
            best_candidate = {
                "gpu_type": gtype,
                "on_demand_hr": cand_price,
                "hbm_gb": cand_vram,
                "peak_bw_tbs": cand_bw,
                "hourly_savings": round(cur_price - cand_price, 3),
                "savings_pct": round((1.0 - cand_price / cur_price) * 100.0, 1),
                "expected_mbu": round(achieved_bw_tbs / cand_bw, 3),
            }
            best_price = cand_price

    return best_candidate
