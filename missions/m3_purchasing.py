"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # --- Extension 1: Advanced Tier Policy & Decision Matrix ---
    advanced_recs = []
    adv_opt_monthly = 0.0
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        jdays = int(num(j.get("days", DAYS)))
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        od_cost = gpu_hours * od

        adv_tier = pricing.recommend_tier_advanced(
            hours_per_day=hpd,
            interruptible=interruptible,
            gpu_type=gtype,
            job_days=jdays,
        )
        if adv_tier == "spot":
            sim = pricing.spot_checkpoint_cost(
                gpu_hours, num(c["spot_hr"]), od,
                interrupt_rate=pricing.DEFAULT_GPU_INTERRUPT_RATES.get(gtype, 0.05)
            )
            cost = sim["spot_cost"]
        elif adv_tier == "reserved":
            # For 24/7 steady production inference -> 3yr reserved; otherwise 1yr reserved
            if hpd >= 24 and not interruptible:
                cost = gpu_hours * num(c["reserved_3yr_hr"])
            else:
                cost = gpu_hours * num(c["reserved_1yr_hr"])
        else:
            cost = od_cost

        adv_opt_monthly += cost
        advanced_recs.append({
            "job_id": j["job_id"], "gpu_type": gtype, "base_tier": recs[len(advanced_recs)]["tier"],
            "adv_tier": adv_tier, "adv_cost": round(cost)
        })

    adv_savings_pct = (on_demand_monthly - adv_opt_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # --- Extension 5: Carbon-Aware Scheduling for Interruptible Workloads ---
    from finops import sustainability
    interruptible_jobs = [j for j in jobs if bool(int(num(j["interruptible"])))]
    total_inter_kwh = sum(
        (num(cat[j["gpu_type"]]["watts"]) * int(num(j["num_gpus"])) * num(j["hours_per_day"]) * DAYS) / 1000.0
        for j in interruptible_jobs
    )

    region_analysis = {}
    base_region = "us-east-1"
    cleanest_region = min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    cheapest_region = min(sustainability.REGION_PRICE_KWH, key=sustainability.REGION_PRICE_KWH.get)

    for reg, gco2 in sustainability.REGION_CARBON.items():
        price_kwh = sustainability.REGION_PRICE_KWH.get(reg, 0.12)
        co2_kg = (total_inter_kwh * gco2) / 1000.0
        elec_usd = total_inter_kwh * price_kwh
        region_analysis[reg] = {
            "gco2_kwh": gco2,
            "usd_kwh": price_kwh,
            "monthly_co2_kg": round(co2_kg, 1),
            "monthly_elec_usd": round(elec_usd, 2),
        }

    co2_saved_cleanest = region_analysis[base_region]["monthly_co2_kg"] - region_analysis[cleanest_region]["monthly_co2_kg"]
    co2_saved_pct = (co2_saved_cleanest / region_analysis[base_region]["monthly_co2_kg"]) * 100

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n== Extension 1: So sánh Chính sách Thu mua Nâng cao ==")
        print("Matrix chính sách: Đánh giá Interruption rate GPU + Kỳ hạn cam kết (1yr vs 3yr):")
        print(f"{'job':18}{'gpu':7}{'Base Tier':11}{'Adv Tier':11}{'Base Opt $':>12}{'Adv Opt $':>12}")
        for ar, br in zip(advanced_recs, recs):
            print(f"{ar['job_id']:18}{ar['gpu_type']:7}{ar['base_tier']:11}{ar['adv_tier']:11}${br['optimized']:>11,}${ar['adv_cost']:>11,}")
        print(f"Savings chính sách cơ bản: {savings_pct:.1f}% (${savings:,.0f}/tháng) | Chính sách nâng cao: {adv_savings_pct:.1f}% (${on_demand_monthly - adv_opt_monthly:,.0f}/tháng)")

        print("\n== Extension 5: Carbon-aware Scheduling cho Workload Interruptible ==")
        print(f"Tổng điện năng tiêu thụ của 5 jobs interruptible: {total_inter_kwh:,.1f} kWh/tháng")
        print(f"{'Region':18}{'$/kWh':>8}{'gCO2/kWh':>10}{'Elec Cost/mo':>14}{'Carbon (kgCO2e)':>16}")
        for reg, ra in sorted(region_analysis.items(), key=lambda x: x[1]["gco2_kwh"]):
            print(f"{reg:18}${ra['usd_kwh']:>7.3f}{ra['gco2_kwh']:>10d}${ra['monthly_elec_usd']:>13.2f}{ra['monthly_co2_kg']:>16.1f}")
        print(f"\n-> Chuyển jobs interruptible từ us-east-1 sang {cleanest_region} (Norway hydro):")
        print(f"   Tiết kiệm {co2_saved_cleanest:,.1f} kgCO2e/tháng ({co2_saved_pct:.1f}% giảm phát thải carbon!)")
        print(f"-> Vùng rẻ nhất: {cheapest_region} ($0.055/kWh, tiết kiệm ${region_analysis[base_region]['monthly_elec_usd'] - region_analysis[cheapest_region]['monthly_elec_usd']:,.2f}/tháng tiền điện)")
        print("-> Đánh giá Latency: europe-north1 có RTT ~120ms sang Mỹ, hoàn toàn phù hợp cho offline training/checkpointing nhưng không dùng cho real-time chat.")

    return {
        "recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
        "extension_policy": {"adv_recs": advanced_recs, "adv_savings_pct": round(adv_savings_pct, 1)},
        "extension_carbon": {
            "total_kwh": round(total_inter_kwh, 1),
            "region_analysis": region_analysis,
            "co2_saved_kg": round(co2_saved_cleanest, 1),
            "cleanest_region": cleanest_region,
            "cheapest_region": cheapest_region,
        },
    }


if __name__ == "__main__":
    run()
