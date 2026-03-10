#!/usr/bin/env python3
"""Shared benchmark database module for kuzco.cpp.

Provides: schema creation, DB helpers, stdlib-only statistics (paired t-test,
normal CDF approximation), telemetry capture, quant type extraction.

Usage:
    import bench_db
    conn = bench_db.connect('data/benchmarks/benchmarks.db')
    bench_db.create_schema(conn)
"""

import math
import os
import re
import sqlite3
import subprocess
import json
from datetime import datetime, timezone

# ── Schema version ──────────────────────────────────────────────────────
SCHEMA_VERSION = 1

# ── Quant type extraction regex ─────────────────────────────────────────
# Matches: IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL,
#          Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K, Q4_K_M, Q5_0, Q5_1, Q5_K_M,
#          Q6_K, Q6_K_L, Q8_0, MXFP4, UD-IQ2_XXS, etc.
QUANT_RE = re.compile(
    r'(UD-)?'
    r'(IQ[1-4]_(?:XXS|XS|S|M|NL)|Q[3-8]_[0-9KL](?:_[SMLK])?|MXFP4)'
)

# ── T-critical table (two-tailed, alpha=0.05) ──────────────────────────
# For paired t-test. df -> t_critical
T_CRITICAL = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    24: 2.064, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000,
    120: 1.980,
}
_T_CRITICAL_KEYS = sorted(T_CRITICAL.keys())
_SQRT2 = math.sqrt(2)


def t_critical(df):
    """Look up or interpolate t-critical value for given degrees of freedom."""
    if df in T_CRITICAL:
        return T_CRITICAL[df]
    if df >= 120:
        return 1.960  # normal approximation
    # Linear interpolation between known values
    for i in range(len(_T_CRITICAL_KEYS) - 1):
        if _T_CRITICAL_KEYS[i] < df < _T_CRITICAL_KEYS[i + 1]:
            lo, hi = _T_CRITICAL_KEYS[i], _T_CRITICAL_KEYS[i + 1]
            frac = (df - lo) / (hi - lo)
            return T_CRITICAL[lo] + frac * (T_CRITICAL[hi] - T_CRITICAL[lo])
    return 1.960  # fallback


def normal_cdf(x):
    """Standard normal CDF via Abramowitz & Stegun 7.1.26 (erf approximation).

    Max error: 7.5e-8. These coefficients approximate erf(z), so we compute
    erf(|x|/√2) and then Φ(x) = 0.5 * (1 + sign * erf(|x|/√2)).
    """
    if x < -8:
        return 0.0
    if x > 8:
        return 1.0
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741,
        -1.453152027, 1.061405429
    )
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    z = abs(x) / _SQRT2
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z)
    return 0.5 * (1.0 + sign * y)


def t_to_pvalue(t_stat, df):
    """Convert t-statistic to two-tailed p-value using normal approximation.

    For df >= 30, uses normal CDF directly. For smaller df, uses the
    approximation: z ≈ t * (1 - 1/(4*df)) / sqrt(1 + t²/(2*df))
    which is reasonably accurate for hypothesis testing.
    """
    if df >= 30:
        return 2 * (1 - normal_cdf(abs(t_stat)))
    # Better approximation for small df
    z = t_stat * (1 - 1 / (4 * df)) / math.sqrt(1 + t_stat ** 2 / (2 * df))
    return 2 * (1 - normal_cdf(abs(z)))


def extract_quant_type(filename):
    """Extract quantization type from model/filename string.

    Examples:
        'Llama-3.2-1B-Instruct-Q4_K_M' -> 'Q4_K_M'
        'Llama-3.2-1B-Instruct-IQ3_XXS' -> 'IQ3_XXS'
        'maverick-128e-iq2xxs' -> 'IQ2_XXS'  (case-insensitive fallback)
    """
    m = QUANT_RE.search(filename)
    if m:
        prefix = m.group(1) or ''
        return prefix + m.group(2)

    # Case-insensitive fallback for filenames like 'iq2xxs'
    lower = filename.lower()
    iq_patterns = [
        ('iq1_m', 'IQ1_M'), ('iq2_xxs', 'IQ2_XXS'), ('iq2_xs', 'IQ2_XS'),
        ('iq2_s', 'IQ2_S'), ('iq3_xxs', 'IQ3_XXS'), ('iq3_s', 'IQ3_S'),
        ('iq4_xs', 'IQ4_XS'), ('iq4_nl', 'IQ4_NL'),
    ]
    # Also try without underscore: 'iq2xxs' -> 'IQ2_XXS'
    for pat, canonical in iq_patterns:
        if pat in lower or pat.replace('_', '') in lower:
            return canonical

    return None


def parse_comment_metadata(line):
    """Parse '# key=value,key=value,...' metadata from CSV comment line.

    Returns dict of key-value pairs.
    """
    if not line.startswith('#'):
        return {}
    content = line[1:].strip()
    meta = {}
    for pair in content.split(','):
        if '=' in pair:
            k, v = pair.split('=', 1)
            meta[k.strip()] = v.strip()
    return meta


# ── Schema DDL ──────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Benchmark database schema v1
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_info (
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS benchmark_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT UNIQUE NOT NULL,
    import_time TEXT NOT NULL DEFAULT (datetime('now')),
    session_date TEXT,
    commit_hash TEXT,
    commit_full TEXT,
    -- Environment
    rocm_version TEXT,
    hip_version TEXT,
    gpu_name TEXT,
    gpu_arch TEXT,
    kernel_version TEXT,
    os_version TEXT,
    cpu_info TEXT,
    gpu_mode TEXT,
    -- Methodology
    n_total_runs INTEGER,
    n_warmup INTEGER DEFAULT 0,
    n_effective INTEGER,
    methodology TEXT,
    -- Extra
    notes TEXT,
    tags TEXT
);

CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    quant_type TEXT,
    metric TEXT NOT NULL DEFAULT 'tg128',
    variant TEXT NOT NULL,
    run_number INTEGER NOT NULL,
    is_warmup INTEGER NOT NULL DEFAULT 0,
    tokens_per_sec REAL NOT NULL,
    section TEXT,
    timestamp TEXT,
    UNIQUE(session_id, model, metric, variant, run_number, is_warmup)
);

CREATE TABLE IF NOT EXISTS system_telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(id) ON DELETE CASCADE,
    captured_at TEXT NOT NULL DEFAULT (datetime('now')),
    capture_phase TEXT DEFAULT 'import',
    -- System load
    load_1m REAL,
    load_5m REAL,
    load_15m REAL,
    -- GPU (rocm-smi)
    gpu_temp_edge_c REAL,
    gpu_power_w REAL,
    gpu_busy_pct REAL,
    gpu_vram_pct REAL,
    gpu_sclk_mhz INTEGER,
    gpu_mclk_mhz INTEGER,
    -- CPU
    cpu_temp_c REAL
);

CREATE TABLE IF NOT EXISTS models (
    model TEXT PRIMARY KEY,
    base_model TEXT,
    quant_type TEXT,
    params_b REAL,
    active_params_b REAL,
    arch_family TEXT,
    n_experts INTEGER,
    n_active INTEGER,
    size_gb REAL,
    bpw REAL,
    -- Validation tiers
    tier1_status TEXT,
    tier2_status TEXT,
    tier3_status TEXT,
    tier4_status TEXT,
    -- Perplexity
    ppl_tmac REAL,
    ppl_stock REAL,
    ppl_delta REAL
);

CREATE TABLE IF NOT EXISTS computed_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES benchmark_sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    metric TEXT NOT NULL DEFAULT 'tg128',
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    -- Sample sizes
    n_pairs INTEGER,
    -- T-MAC stats
    tmac_mean REAL,
    tmac_sd REAL,
    tmac_ci95_lo REAL,
    tmac_ci95_hi REAL,
    -- Stock stats
    stock_mean REAL,
    stock_sd REAL,
    stock_ci95_lo REAL,
    stock_ci95_hi REAL,
    -- Speedup
    speedup_pct REAL,
    speedup_ci95_lo REAL,
    speedup_ci95_hi REAL,
    -- Significance
    t_statistic REAL,
    p_value REAL,
    is_significant INTEGER,
    -- CV
    tmac_cv_pct REAL,
    stock_cv_pct REAL,
    UNIQUE(session_id, model, metric)
);

"""


def connect(db_path):
    """Connect to benchmark database, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_schema(conn):
    """Create all tables, views, and insert schema version."""
    conn.executescript(SCHEMA_SQL)
    # Insert schema version if not present
    existing = conn.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
    if not existing:
        conn.execute("INSERT INTO schema_info (version) VALUES (?)", (SCHEMA_VERSION,))
    conn.commit()


def get_schema_version(conn):
    """Return current schema version, or 0 if not initialized."""
    try:
        row = conn.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
        return row['version'] if row else 0
    except sqlite3.OperationalError:
        return 0


# ── Statistics ──────────────────────────────────────────────────────────

def paired_ttest(tmac_values, stock_values):
    """Compute paired t-test statistics from two matched lists.

    Returns dict with all computed_stats fields, or None if insufficient data.
    """
    n = len(tmac_values)
    if n != len(stock_values) or n < 2:
        return None

    # Differences (tmac - stock)
    diffs = [t - s for t, s in zip(tmac_values, stock_values)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    sd_diff = math.sqrt(var_diff) if var_diff > 0 else 0

    # T-MAC descriptive stats
    tmac_mean = sum(tmac_values) / n
    tmac_var = sum((v - tmac_mean) ** 2 for v in tmac_values) / (n - 1) if n > 1 else 0
    tmac_sd = math.sqrt(tmac_var)

    # Stock descriptive stats
    stock_mean = sum(stock_values) / n
    stock_var = sum((v - stock_mean) ** 2 for v in stock_values) / (n - 1) if n > 1 else 0
    stock_sd = math.sqrt(stock_var)

    # t-statistic
    se = sd_diff / math.sqrt(n) if sd_diff > 0 else 0
    t_stat = mean_diff / se if se > 0 else 0
    df = n - 1

    # p-value
    p_val = t_to_pvalue(t_stat, df)

    # Confidence intervals
    tc = t_critical(df)

    # Speedup
    speedup_pct = 100 * (tmac_mean - stock_mean) / stock_mean if stock_mean > 0 else 0

    # Speedup CI via ratio method
    # For each pair, compute per-run speedup ratio
    ratios = [100 * (t - s) / s for t, s in zip(tmac_values, stock_values) if s > 0]
    if len(ratios) >= 2:
        ratio_mean = sum(ratios) / len(ratios)
        ratio_var = sum((r - ratio_mean) ** 2 for r in ratios) / (len(ratios) - 1)
        ratio_se = math.sqrt(ratio_var / len(ratios))
        speedup_ci_lo = ratio_mean - tc * ratio_se
        speedup_ci_hi = ratio_mean + tc * ratio_se
    else:
        speedup_ci_lo = speedup_pct
        speedup_ci_hi = speedup_pct

    # CVs
    tmac_cv = 100 * tmac_sd / tmac_mean if tmac_mean > 0 else 0
    stock_cv = 100 * stock_sd / stock_mean if stock_mean > 0 else 0

    # Mean CIs
    tmac_se = tmac_sd / math.sqrt(n) if n > 0 else 0
    stock_se = stock_sd / math.sqrt(n) if n > 0 else 0

    return {
        'n_pairs': n,
        'tmac_mean': round(tmac_mean, 6),
        'tmac_sd': round(tmac_sd, 6),
        'tmac_ci95_lo': round(tmac_mean - tc * tmac_se, 6),
        'tmac_ci95_hi': round(tmac_mean + tc * tmac_se, 6),
        'stock_mean': round(stock_mean, 6),
        'stock_sd': round(stock_sd, 6),
        'stock_ci95_lo': round(stock_mean - tc * stock_se, 6),
        'stock_ci95_hi': round(stock_mean + tc * stock_se, 6),
        'speedup_pct': round(speedup_pct, 2),
        'speedup_ci95_lo': round(speedup_ci_lo, 2),
        'speedup_ci95_hi': round(speedup_ci_hi, 2),
        't_statistic': round(t_stat, 4),
        'p_value': round(p_val, 6),
        'is_significant': 1 if p_val < 0.05 else 0,
        'tmac_cv_pct': round(tmac_cv, 4),
        'stock_cv_pct': round(stock_cv, 4),
    }


# ── Telemetry capture ──────────────────────────────────────────────────

def capture_telemetry():
    """Capture current system telemetry. Best-effort, all fields nullable."""
    tel = {}

    # /proc/loadavg
    try:
        with open('/proc/loadavg') as f:
            parts = f.read().split()
            tel['load_1m'] = float(parts[0])
            tel['load_5m'] = float(parts[1])
            tel['load_15m'] = float(parts[2])
    except (OSError, IndexError, ValueError):
        pass

    # rocm-smi --json
    try:
        result = subprocess.run(
            ['rocm-smi', '--json', '--showtemp', '--showpower',
             '--showuse', '--showmemuse', '--showclocks'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # rocm-smi JSON structure varies; try common keys
            for card_key in ['card0', 'GPU[0]', '0']:
                card = data.get(card_key, {})
                if card:
                    for k, v in card.items():
                        kl = k.lower()
                        try:
                            if 'temperature' in kl and 'edge' in kl:
                                tel['gpu_temp_edge_c'] = float(str(v).rstrip(' C°'))
                            elif 'power' in kl and 'average' in kl:
                                tel['gpu_power_w'] = float(str(v).rstrip(' W'))
                            elif 'gpu use' in kl or 'gpu_busy' in kl:
                                tel['gpu_busy_pct'] = float(str(v).rstrip(' %'))
                            elif 'vram' in kl and 'use' in kl:
                                tel['gpu_vram_pct'] = float(str(v).rstrip(' %'))
                            elif 'sclk' in kl:
                                val = str(v).rstrip('Mhz ')
                                tel['gpu_sclk_mhz'] = int(float(val))
                            elif 'mclk' in kl:
                                val = str(v).rstrip('Mhz ')
                                tel['gpu_mclk_mhz'] = int(float(val))
                        except (ValueError, TypeError):
                            continue
                    break
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    # CPU temp via sensors
    try:
        result = subprocess.run(
            ['sensors', '-j'], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for chip_name, chip_data in data.items():
                if 'k10temp' in chip_name or 'zenpower' in chip_name:
                    for sensor_name, sensor_data in chip_data.items():
                        if isinstance(sensor_data, dict):
                            for k, v in sensor_data.items():
                                if 'input' in k.lower() and 'tctl' in sensor_name.lower():
                                    tel['cpu_temp_c'] = float(v)
                                    break
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    return tel


def insert_telemetry(conn, session_id, phase='import'):
    """Capture and insert telemetry for a session. Returns telemetry dict."""
    tel = capture_telemetry()
    if not tel:
        return tel

    conn.execute("""
        INSERT INTO system_telemetry
            (session_id, capture_phase, load_1m, load_5m, load_15m,
             gpu_temp_edge_c, gpu_power_w, gpu_busy_pct, gpu_vram_pct,
             gpu_sclk_mhz, gpu_mclk_mhz, cpu_temp_c)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id, phase,
        tel.get('load_1m'), tel.get('load_5m'), tel.get('load_15m'),
        tel.get('gpu_temp_edge_c'), tel.get('gpu_power_w'),
        tel.get('gpu_busy_pct'), tel.get('gpu_vram_pct'),
        tel.get('gpu_sclk_mhz'), tel.get('gpu_mclk_mhz'),
        tel.get('cpu_temp_c'),
    ))
    conn.commit()
    return tel


# ── Chart data queries ──────────────────────────────────────────────────

def query_chart_q4km(conn, gpu_mode='single'):
    """Query data for Q4_K_M speedup chart."""
    rows = conn.execute("""
        SELECT cs.model, m.base_model, m.arch_family,
               cs.tmac_mean, cs.stock_mean, cs.speedup_pct,
               cs.speedup_ci95_lo, cs.speedup_ci95_hi,
               cs.n_pairs, cs.p_value, bs.gpu_mode
        FROM computed_stats cs
        JOIN benchmark_sessions bs ON cs.session_id = bs.id
        LEFT JOIN models m ON cs.model = m.model
        WHERE cs.metric = 'tg128'
          AND COALESCE(bs.gpu_mode, 'single') = ?
          AND (m.quant_type = 'Q4_K_M' OR cs.model LIKE '%Q4_K_M%')
        ORDER BY cs.speedup_pct ASC
    """, (gpu_mode,)).fetchall()
    return [dict(r) for r in rows]


def query_chart_iq(conn):
    """Query data for IQ types speedup chart."""
    rows = conn.execute("""
        SELECT cs.model, m.base_model, m.arch_family, m.bpw,
               cs.tmac_mean, cs.stock_mean, cs.speedup_pct,
               cs.speedup_ci95_lo, cs.speedup_ci95_hi,
               cs.n_pairs, cs.p_value, bs.gpu_mode
        FROM computed_stats cs
        JOIN benchmark_sessions bs ON cs.session_id = bs.id
        LEFT JOIN models m ON cs.model = m.model
        WHERE cs.metric = 'tg128'
          AND (m.quant_type LIKE 'IQ%' OR cs.model LIKE '%IQ%')
        ORDER BY cs.speedup_pct ASC
    """).fetchall()
    return [dict(r) for r in rows]


def query_chart_multigpu(conn):
    """Query data for multi-GPU throughput chart."""
    rows = conn.execute("""
        SELECT cs.model, m.base_model, m.arch_family,
               cs.tmac_mean, cs.stock_mean, cs.speedup_pct,
               cs.speedup_ci95_lo, cs.speedup_ci95_hi,
               cs.n_pairs, bs.gpu_mode
        FROM computed_stats cs
        JOIN benchmark_sessions bs ON cs.session_id = bs.id
        LEFT JOIN models m ON cs.model = m.model
        WHERE cs.metric = 'tg128'
          AND bs.gpu_mode = 'dual'
        ORDER BY cs.speedup_pct DESC
    """).fetchall()
    return [dict(r) for r in rows]


def query_chart_absolute(conn, gpu_mode='single'):
    """Query data for absolute throughput chart (stock vs T-MAC bars)."""
    rows = conn.execute("""
        SELECT cs.model, m.base_model, m.arch_family,
               cs.tmac_mean, cs.stock_mean, cs.speedup_pct,
               cs.n_pairs, bs.gpu_mode
        FROM computed_stats cs
        JOIN benchmark_sessions bs ON cs.session_id = bs.id
        LEFT JOIN models m ON cs.model = m.model
        WHERE cs.metric = 'tg128'
          AND COALESCE(bs.gpu_mode, 'single') = ?
        ORDER BY cs.stock_mean DESC
    """, (gpu_mode,)).fetchall()
    return [dict(r) for r in rows]
