"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- Extension 2: Right-sizing according to MBU and VRAM unit economics ---
    catalog_raw = {k: {col: num(val) for col, val in v.items()} for k, v in cat.items()}
    unit_economics = {}
    for gtype, row in catalog_raw.items():
        unit_economics[gtype] = {
            "on_demand_hr": row["on_demand_hr"],
            "hbm_gb": row["hbm_gb"],
            "peak_bw_tbs": row["peak_bw_tbs"],
            "cost_per_gb_vram": metrics.gpu_cost_per_vram_gb(row["on_demand_hr"], row["hbm_gb"]),
            "cost_per_tbs": metrics.gpu_cost_per_bandwidth_tbs(row["on_demand_hr"], row["peak_bw_tbs"]),
        }

    # Evaluate right-sizing for memory-bound / low-MBU GPUs (MBU < 0.35)
    mbu_rightsizing = []
    total_rightsize_monthly = 0.0
    for s in summary:
        if s["mbu"] < 0.35 and s["gpu_util_pct"] >= 20.0:
            cur_bw_achieved = s["mbu"] * num(cat[s["gpu_type"]]["peak_bw_tbs"])
            cand = metrics.find_mbu_rightsize_candidate(
                current_gpu=s["gpu_type"],
                achieved_bw_tbs=cur_bw_achieved,
                current_mbu=s["mbu"],
                catalog=catalog_raw,
            )
            if cand:
                monthly_saved = cand["hourly_savings"] * 24 * 30
                total_rightsize_monthly += monthly_saved
                mbu_rightsizing.append({
                    "gpu_id": s["gpu_id"],
                    "current_type": s["gpu_type"],
                    "current_mbu": s["mbu"],
                    "achieved_bw": round(cur_bw_achieved, 3),
                    "proposed_type": cand["gpu_type"],
                    "proposed_bw": cand["peak_bw_tbs"],
                    "proposed_mbu": cand["expected_mbu"],
                    "hourly_savings": cand["hourly_savings"],
                    "savings_pct": cand["savings_pct"],
                    "monthly_savings": round(monthly_saved),
                    "cur_cost_per_gb": round(unit_economics[s["gpu_type"]]["cost_per_gb_vram"], 4),
                    "prop_cost_per_gb": round(unit_economics[cand["gpu_type"]]["cost_per_gb_vram"], 4),
                })

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        # Extension 2 detailed reporting
        print("\n== Extension 2: Right-sizing theo MBU & VRAM Economics ==")
        print("Bảng đơn giá VRAM và băng thông theo GPU catalog:")
        print(f"{'GPU':8}{'$/hr':>7}{'VRAM(GB)':>10}{'BW(TB/s)':>10}{'$/GB-VRAM/h':>14}{'$/(TB/s)/h':>13}")
        for gtype, u in sorted(unit_economics.items(), key=lambda x: x[1]["cost_per_gb_vram"]):
            print(f"{gtype:8}${u['on_demand_hr']:>6.2f}{u['hbm_gb']:>10.0f}{u['peak_bw_tbs']:>10.2f}${u['cost_per_gb_vram']:>13.4f}${u['cost_per_tbs']:>12.2f}")

        print("\nĐề xuất Right-sizing cho các GPU memory-bound (MBU thấp):")
        print(f"{'GPU ID':12}{'Current':9}{'MBU':>6}{'Proposed':10}{'Prop BW':>9}{'Savings/mo':>12}{'Save%':>7}")
        for r in mbu_rightsizing:
            print(f"{r['gpu_id']:12}{r['current_type']:9}{r['current_mbu']:>6.3f}{r['proposed_type']:10}{r['proposed_bw']:>8.1f}T${r['monthly_savings']:>10,}{r['savings_pct']:>6.1f}%")
        print(f"Tổng tiết kiệm hàng tháng từ right-sizing theo MBU: ${total_rightsize_monthly:,.0f}/tháng")
        print("Insight: Không chỉ chọn GPU rẻ nhất ($/GPU-hr) vì workload decode bị nghẽn bởi memory bandwidth (HBM) và dung lượng VRAM (chứa KV-cache); nếu ép xuống L4 (0.3 TB/s, 24GB) sẽ gây nghẽn nghiêm trọng hoặc OOM!")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "mbu_rightsizing": mbu_rightsizing,
        "mbu_monthly_savings": round(total_rightsize_monthly),
    }


if __name__ == "__main__":
    run()
