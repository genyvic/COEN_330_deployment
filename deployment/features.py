"""
features.py - turns 5 seconds of measurements into the 23 model inputs.

This follows the training preprocessing notebook
(COEN_330_Project_Data_Preprocessing.ipynb) step by step:
  - each perf counter is measured once per second (s1..s5), then averaged
  - CPU frequency of each second = cycles / (task-clock in ms * 1000), then averaged
  - 4 rates are made from the averages (IPC, cache miss rate, ...)

No Linux-only code here, so the website and the tests can use it on any computer.
"""

# The 23 inputs, in the same order as the training table (dataset_B_final.csv)
FEATURES = [
    "cpu_temp_c", "idle_power_w", "background_cpu_pct", "threads_max",
    "mem_rss_max_mb", "io_read_chars", "io_write_chars",
    "cycles_mean", "instructions_mean", "cache_misses_mean", "cache_references_mean",
    "context_switches_mean", "page_faults_mean", "task_clock_mean", "cpu_migrations_mean",
    "major_faults_mean", "branch_misses_mean", "l1_dcache_load_misses_mean",
    "cpu_freq_mhz_mean", "instructions_per_cycle", "cache_miss_rate",
    "context_switch_rate", "page_fault_rate",
]

# perf counters that are averaged over the 5 seconds (column names, not perf names)
MEAN_COUNTERS = [
    "cycles", "instructions", "cache_misses", "cache_references", "context_switches",
    "page_faults", "task_clock", "cpu_migrations", "major_faults", "branch_misses",
    "l1_dcache_load_misses",
]

WINDOW_S = 5
TINY = 1e-10   # same small number as the notebook, to avoid dividing by 0


def _mean(values):
    """Mean that skips missing values (like pandas .mean(axis=1))."""
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _freq_mhz(cycles, task_clock_ms):
    """Same formula as collect_data.py: cycles / CPU time."""
    if cycles is None or not task_clock_ms:
        return None
    return cycles / (task_clock_ms * 1000)


def compute_features(raw):
    """raw = one measurement, with the same names as the collected data:
         raw["per_sec"]  = {"cycles": [s1, s2, s3, s4, s5], "task_clock": [...], ...}
         raw["cpu_temp_c"], raw["idle_power_w"], raw["background_cpu_pct"],
         raw["threads_max"], raw["mem_rss_max_mb"], raw["io_read_chars"], raw["io_write_chars"]
       Returns {feature name: value}. A value is None if it couldn't be measured."""
    per_sec = raw["per_sec"]
    f = {k: raw.get(k) for k in ["cpu_temp_c", "idle_power_w", "background_cpu_pct",
                                 "threads_max", "mem_rss_max_mb", "io_read_chars",
                                 "io_write_chars"]}

    for name in MEAN_COUNTERS:
        f[f"{name}_mean"] = _mean(per_sec.get(name, [None] * WINDOW_S))

    freqs = [_freq_mhz(c, t) for c, t in zip(per_sec.get("cycles", [None] * WINDOW_S),
                                             per_sec.get("task_clock", [None] * WINDOW_S))]
    f["cpu_freq_mhz_mean"] = _mean(freqs)

    def rate(a, b):
        if f[a] is None or f[b] is None:
            return None
        return f[a] / (f[b] + TINY)

    f["instructions_per_cycle"] = rate("instructions_mean", "cycles_mean")
    f["cache_miss_rate"] = rate("cache_misses_mean", "cache_references_mean")
    f["context_switch_rate"] = rate("context_switches_mean", "task_clock_mean")
    f["page_fault_rate"] = rate("page_faults_mean", "task_clock_mean")
    return {k: f[k] for k in FEATURES}


def raw_from_collected_row(row):
    """Builds `raw` from one row of the collected CSV files (combined_test.csv ...).
    Used by verify.py to check that compute_features() matches the notebook."""
    def num(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if v != v else v          # NaN -> None

    per_sec = {c: [num(row.get(f"{c}_s{i}")) for i in range(1, WINDOW_S + 1)]
               for c in MEAN_COUNTERS}
    raw = {k: num(row.get(k)) for k in ["cpu_temp_c", "idle_power_w", "background_cpu_pct",
                                        "threads_max", "mem_rss_max_mb", "io_read_chars",
                                        "io_write_chars"]}
    raw["per_sec"] = per_sec
    return raw
