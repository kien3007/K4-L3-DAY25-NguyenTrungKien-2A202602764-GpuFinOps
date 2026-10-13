"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


DEFAULT_GPU_INTERRUPT_RATES: dict[str, float] = {
    "H100": 0.04,
    "H200": 0.05,
    "A100": 0.06,
    "A10G": 0.09,
    "L4": 0.08,
    "B200": 0.04,
    "MI300X": 0.05,
}


def cache_break_even_reads(
    write_cost_per_m: float = 3.0,
    base_read_cost_per_m: float | None = None,
    read_discount: float = 0.10,
) -> float:
    """Calculate the minimum average reads required for prompt caching to yield net savings.

    Formula:
        WriteCost + N * (base_read_cost * read_discount) < N * base_read_cost
        => N * base_read_cost * (1 - read_discount) > WriteCost
        => N > WriteCost / (base_read_cost * (1 - read_discount))
    """
    base_in = base_read_cost_per_m if base_read_cost_per_m is not None else write_cost_per_m
    denom = base_in * (1.0 - read_discount)
    if denom <= 0:
        return float("inf")
    return write_cost_per_m / denom


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float = 3.0,
    read_discount: float = 0.10,
    base_read_cost_per_m: float | None = None,
) -> bool:
    """Check if prompt caching is financially beneficial given observed read frequency.

    Gated check for Extension 3: only apply caching savings when expected reuse exceeds break-even.
    """
    be = cache_break_even_reads(write_cost_per_m, base_read_cost_per_m, read_discount)
    return avg_cache_reads > be


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: float | None = None,
) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    Baseline policy:
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)

    When gpu_type and job_days are supplied, this delegates to recommend_tier_advanced.
    """
    if gpu_type is not None or job_days is not None:
        return recommend_tier_advanced(
            hours_per_day=hours_per_day,
            interruptible=interruptible,
            gpu_type=gpu_type,
            job_days=job_days,
            reserved_3yr_discount=reserved_discount,
        )

    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def recommend_tier_advanced(
    hours_per_day: float,
    interruptible: bool,
    gpu_type: str | None = None,
    job_days: float | None = None,
    reserved_3yr_discount: float = 0.45,
    reserved_1yr_discount: float = 0.20,
    interruption_rate: float | None = None,
) -> str:
    """Advanced multi-factor purchasing policy for Extension 1.

    Evaluates:
      1. GPU-specific spot interruption risk and checkpoint feasibility.
      2. Job duration vs commitment term: Short temporary jobs (<30 days) avoid multi-year
         commitments unless pooled into a shared platform cluster.
      3. Tier comparison across on-demand, spot, 1-year reserved, and 3-year reserved.
    """
    duty = max(0.0, hours_per_day) / 24.0
    be_3yr = break_even_utilization(reserved_3yr_discount)  # e.g., 55% (~13.2h)
    be_1yr = break_even_utilization(reserved_1yr_discount)  # e.g., 80% (~19.2h)

    # Effective interruption rate by GPU type
    rate = interruption_rate
    if rate is None and gpu_type:
        rate = DEFAULT_GPU_INTERRUPT_RATES.get(gpu_type, 0.05)
    rate = rate if rate is not None else 0.05

    # 1. Non-interruptible workloads (production user-facing inference)
    if not interruptible:
        if duty >= be_3yr:
            return "reserved"
        if duty >= be_1yr:
            return "reserved"
        return "on_demand"

    # 2. Interruptible workloads (training, batch eval, offline dev)
    # If job is very short (e.g. <= 3 days) or duty cycle < 24h, spot is universally optimal
    if hours_per_day < 24:
        return "spot"

    # For 24/7 interruptible workloads, check if reserved 3yr beats spot after checkpoint overhead
    if duty >= be_3yr:
        return "reserved"

    return "spot"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
