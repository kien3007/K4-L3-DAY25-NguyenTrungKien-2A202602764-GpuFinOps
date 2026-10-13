import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finops import pricing, metrics, sustainability


def test_recommend_tier_advanced():
    # 24/7 steady inference -> reserved
    assert pricing.recommend_tier_advanced(24, False, gpu_type="A10G", job_days=30) == "reserved"
    # Short interruptible training -> spot
    assert pricing.recommend_tier_advanced(10, True, gpu_type="A100", job_days=5) == "spot"
    # Low duty non-interruptible -> on_demand
    assert pricing.recommend_tier_advanced(4, False, gpu_type="H100", job_days=10) == "on_demand"


def test_cache_is_worth_it_logic():
    be = pricing.cache_break_even_reads(write_cost_per_m=3.0, base_read_cost_per_m=3.0, read_discount=0.10)
    assert abs(be - (1.0 / 0.90)) < 1e-4

    # Gating check
    assert pricing.cache_is_worth_it(avg_cache_reads=3.5, write_cost_per_m=3.0, read_discount=0.10) is True
    assert pricing.cache_is_worth_it(avg_cache_reads=0.8, write_cost_per_m=3.0, read_discount=0.10) is False


def test_mbu_rightsizing_logic():
    vram_cost = metrics.gpu_cost_per_vram_gb(on_demand_hr=2.50, hbm_gb=80.0)
    assert abs(vram_cost - 0.03125) < 1e-6

    bw_cost = metrics.gpu_cost_per_bandwidth_tbs(on_demand_hr=2.50, peak_bw_tbs=3.35)
    assert bw_cost > 0.70

    catalog = {
        "H100": {"on_demand_hr": 2.50, "hbm_gb": 80.0, "peak_bw_tbs": 3.35},
        "A100": {"on_demand_hr": 1.79, "hbm_gb": 80.0, "peak_bw_tbs": 2.00},
        "L4": {"on_demand_hr": 0.80, "hbm_gb": 24.0, "peak_bw_tbs": 0.30},
    }
    # An H100 achieving only 0.69 TB/s can be safely right-sized to A100 (2.0 TB/s headroom)
    cand = metrics.find_mbu_rightsize_candidate(
        current_gpu="H100",
        achieved_bw_tbs=0.69,
        current_mbu=0.207,
        catalog=catalog,
        mbu_headroom=0.60,
    )
    assert cand is not None
    assert cand["gpu_type"] == "A100"
    assert cand["hourly_savings"] > 0


def test_carbon_aware_savings():
    kwh = 1000.0
    carbon_us = sustainability.carbon_g(kwh * 1000.0, region="us-east-1")
    carbon_nordic = sustainability.carbon_g(kwh * 1000.0, region="europe-north1")
    assert carbon_us > carbon_nordic
    # Europe-north1 has 30 g/kWh vs us-east-1 380 g/kWh (>90% reduction)
    assert (carbon_us - carbon_nordic) / carbon_us > 0.90
