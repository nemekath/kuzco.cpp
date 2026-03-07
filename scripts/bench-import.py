#!/usr/bin/env python3
"""Benchmark import tool for kuzco.cpp.

CLI tool that bridges CSV benchmark data into the SQLite database.

Usage:
    bench-import.py init [--force]
    bench-import.py import <csv> [--telemetry] [--model NAME] [--warmup N]
    bench-import.py import-dir <dir> [--telemetry]
    bench-import.py import-zoo <tracker.md>
    bench-import.py recompute [--session-id N]
    bench-import.py enrich --config <json>
    bench-import.py status [--model NAME]
    bench-import.py export-chart-data <name>
    bench-import.py export-table <name> [--format md|csv]
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

# Add scripts dir to path for bench_db import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_db

DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'data', 'benchmarks', 'benchmarks.db'
)


# ── CSV Reading Helpers ────────────────────────────────────────────────

def read_csv_file(filepath):
    """Read a CSV file once, returning (lines, metadata, format).

    Splits the file into comment metadata, detects format from header,
    and returns all lines for parsing.
    """
    with open(filepath, 'r') as f:
        all_lines = f.readlines()

    meta = {}
    fmt = None

    for line in all_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            meta.update(bench_db.parse_comment_metadata(stripped))
            continue
        # First non-comment, non-empty line is the header
        cols = [c.strip().lower() for c in stripped.split(',')]
        if 'section' in cols and 'tokens_per_sec' in cols:
            fmt = 'A'
        elif 'tier' in cols and 'unit' in cols:
            fmt = 'B'
        elif 'type' in cols and 'avg_ts' in cols:
            fmt = 'C'
        break

    return all_lines, meta, fmt


def _session_from_meta(meta, max_run, **overrides):
    """Build a session dict from CSV comment metadata."""
    s = {
        'session_date': meta.get('date'),
        'commit_hash': meta.get('commit'),
        'rocm_version': meta.get('rocm'),
        'hip_version': meta.get('hip'),
        'gpu_name': meta.get('gpu'),
        'kernel_version': meta.get('kernel'),
        'gpu_mode': meta.get('gpu_mode', 'single'),
        'n_total_runs': max_run,
        'n_warmup': 0,
        'n_effective': max_run,
        'methodology': 'paired_interleaved',
    }
    s.update(overrides)
    return s


# ── Format A Parser ────────────────────────────────────────────────────
# section,model,metric,run,variant,tokens_per_sec

def parse_format_a(lines, meta):
    """Parse Format A CSV. Returns (session_data, measurements)."""
    rows = []
    reader = csv.DictReader(
        (line for line in lines if not line.startswith('#')),
    )
    for row in reader:
        rows.append({
            'model': row['model'].strip(),
            'quant_type': bench_db.extract_quant_type(row['model']),
            'metric': row.get('metric', 'tg128').strip(),
            'variant': row['variant'].strip(),
            'run_number': int(row['run']),
            'is_warmup': 0,
            'tokens_per_sec': float(row['tokens_per_sec']),
            'section': row.get('section', '').strip(),
            'timestamp': None,
        })

    max_run = max((r['run_number'] for r in rows), default=0)
    return _session_from_meta(meta, max_run), rows


# ── Format B Parser ────────────────────────────────────────────────────
# tier,model,metric,run,variant,value,unit

def parse_format_b(lines, meta):
    """Parse Format B CSV (tier-based). Returns (session_data, measurements)."""
    rows = []
    reader = csv.DictReader(
        (line for line in lines if not line.startswith('#')),
    )
    for row in reader:
            rows.append({
                'model': row['model'].strip(),
                'quant_type': bench_db.extract_quant_type(row['model']),
                'metric': row.get('metric', 'tg128').strip(),
                'variant': row['variant'].strip(),
                'run_number': int(row['run']),
                'is_warmup': 0,
                'tokens_per_sec': float(row['value']),
                'section': row.get('tier', '').strip(),
                'timestamp': None,
            })

    max_run = max((r['run_number'] for r in rows), default=0)
    return _session_from_meta(meta, max_run), rows


# ── Format C Parser ────────────────────────────────────────────────────
# run,type,avg_ts[,stddev_ts],timestamp

def parse_format_c(lines, meta, filename=''):
    """Parse Format C CSV (compact). Returns (session_data, measurements).

    Model name must come from metadata comment or --model flag.
    """
    rows = []
    model_name = meta.get('model', os.path.basename(filename).rsplit('.', 1)[0])

    reader = csv.DictReader(
        (line for line in lines if not line.startswith('#')),
    )
    for row in reader:
        ts_val = row.get('avg_ts', '0').strip().strip('"')

        rows.append({
            'model': model_name,
            'quant_type': bench_db.extract_quant_type(model_name),
            'metric': 'tg128',
            'variant': row['type'].strip(),
            'run_number': int(row['run']),
            'is_warmup': 0,
            'tokens_per_sec': float(ts_val),
            'section': '',
            'timestamp': row.get('timestamp', '').strip() or None,
        })

    max_run = max((r['run_number'] for r in rows), default=0)
    return _session_from_meta(meta, max_run,
                              gpu_mode=meta.get('gpu_mode'),
                              notes=meta.get('ngl', '')), rows


# ── Import Logic ────────────────────────────────────────────────────────

def import_csv(conn, filepath, telemetry=False, model_override=None, warmup=0):
    """Import a single CSV file into the database.

    Returns session_id on success, None if already imported.
    """
    source_file = os.path.basename(filepath)

    # Check dedup
    existing = conn.execute(
        "SELECT id FROM benchmark_sessions WHERE source_file = ?",
        (source_file,)
    ).fetchone()
    if existing:
        print(f"  SKIP (already imported): {source_file}")
        return None

    # Read file once, detect format + metadata
    lines, meta, fmt = read_csv_file(filepath)
    if not fmt:
        print(f"  ERROR: Unknown CSV format: {source_file}")
        return None

    if model_override:
        meta['model'] = model_override

    # Parse
    if fmt == 'C':
        session_data, measurements = parse_format_c(lines, meta, filename=filepath)
    else:
        parsers = {'A': parse_format_a, 'B': parse_format_b}
        session_data, measurements = parsers[fmt](lines, meta)

    if not measurements:
        print(f"  ERROR: No measurements found in {source_file}")
        return None

    # Apply warmup
    if warmup > 0:
        for m in measurements:
            if m['run_number'] <= warmup:
                m['is_warmup'] = 1
        session_data['n_warmup'] = warmup
        session_data['n_effective'] = session_data['n_total_runs'] - warmup

    # Insert session
    session_data['source_file'] = source_file
    cols = list(session_data.keys())
    placeholders = ', '.join(['?'] * len(cols))
    col_names = ', '.join(cols)

    cursor = conn.execute(
        f"INSERT INTO benchmark_sessions ({col_names}) VALUES ({placeholders})",
        [session_data[c] for c in cols]
    )
    session_id = cursor.lastrowid

    # Insert measurements (bulk)
    conn.executemany("""
        INSERT INTO measurements
            (session_id, model, quant_type, metric, variant,
             run_number, is_warmup, tokens_per_sec, section, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (session_id, m['model'], m['quant_type'], m['metric'],
         m['variant'], m['run_number'], m['is_warmup'],
         m['tokens_per_sec'], m['section'], m['timestamp'])
        for m in measurements
    ])

    conn.commit()

    # Telemetry
    if telemetry:
        tel = bench_db.insert_telemetry(conn, session_id)
        if tel:
            print(f"  Telemetry captured: load={tel.get('load_1m', '?')}, "
                  f"GPU={tel.get('gpu_temp_edge_c', '?')}°C")

    n_meas = len(measurements)
    n_models = len(set(m['model'] for m in measurements))
    print(f"  OK format={fmt}: {source_file} → {n_meas} measurements, "
          f"{n_models} model(s), session #{session_id}")

    return session_id


# ── Recompute Stats ────────────────────────────────────────────────────

def recompute_stats(conn, session_id=None):
    """Recompute paired t-test statistics from raw measurements.

    If session_id is given, only recompute for that session.
    """
    if session_id:
        filter_clause = "WHERE m.session_id = ? AND m.is_warmup = 0"
        params = (session_id,)
    else:
        filter_clause = "WHERE m.is_warmup = 0"
        params = ()

    # Fetch all non-warmup measurements in one query, grouped by key
    all_rows = conn.execute(f"""
        SELECT m.session_id, m.model, m.metric, m.variant, m.tokens_per_sec
        FROM measurements m
        {filter_clause}
        ORDER BY m.session_id, m.model, m.metric, m.variant, m.run_number
    """, params).fetchall()

    # Group by (session_id, model, metric, variant)
    groups = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        key = (r['session_id'], r['model'], r['metric'])
        groups[key][r['variant']].append(r['tokens_per_sec'])

    computed = 0
    for (sid, model, metric), variants in groups.items():
        tmac_vals = variants.get('tmac', [])
        stock_vals = variants.get('stock', [])

        if not tmac_vals or not stock_vals:
            continue

        # Truncate to same length (pair matching)
        n = min(len(tmac_vals), len(stock_vals))
        tmac_vals = tmac_vals[:n]
        stock_vals = stock_vals[:n]

        stats = bench_db.paired_ttest(tmac_vals, stock_vals)
        if not stats:
            continue

        # Upsert computed_stats
        conn.execute("""
            INSERT INTO computed_stats
                (session_id, model, metric, n_pairs,
                 tmac_mean, tmac_sd, tmac_ci95_lo, tmac_ci95_hi,
                 stock_mean, stock_sd, stock_ci95_lo, stock_ci95_hi,
                 speedup_pct, speedup_ci95_lo, speedup_ci95_hi,
                 t_statistic, p_value, is_significant,
                 tmac_cv_pct, stock_cv_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, model, metric) DO UPDATE SET
                computed_at = datetime('now'),
                n_pairs = excluded.n_pairs,
                tmac_mean = excluded.tmac_mean,
                tmac_sd = excluded.tmac_sd,
                tmac_ci95_lo = excluded.tmac_ci95_lo,
                tmac_ci95_hi = excluded.tmac_ci95_hi,
                stock_mean = excluded.stock_mean,
                stock_sd = excluded.stock_sd,
                stock_ci95_lo = excluded.stock_ci95_lo,
                stock_ci95_hi = excluded.stock_ci95_hi,
                speedup_pct = excluded.speedup_pct,
                speedup_ci95_lo = excluded.speedup_ci95_lo,
                speedup_ci95_hi = excluded.speedup_ci95_hi,
                t_statistic = excluded.t_statistic,
                p_value = excluded.p_value,
                is_significant = excluded.is_significant,
                tmac_cv_pct = excluded.tmac_cv_pct,
                stock_cv_pct = excluded.stock_cv_pct
        """, (
            sid, model, metric, stats['n_pairs'],
            stats['tmac_mean'], stats['tmac_sd'],
            stats['tmac_ci95_lo'], stats['tmac_ci95_hi'],
            stats['stock_mean'], stats['stock_sd'],
            stats['stock_ci95_lo'], stats['stock_ci95_hi'],
            stats['speedup_pct'], stats['speedup_ci95_lo'], stats['speedup_ci95_hi'],
            stats['t_statistic'], stats['p_value'], stats['is_significant'],
            stats['tmac_cv_pct'], stats['stock_cv_pct'],
        ))
        computed += 1

    conn.commit()
    return computed


# ── Import Zoo ─────────────────────────────────────────────────────────

def import_zoo(conn, tracker_path):
    """Import model metadata from model-zoo-tracker.md.

    Parses markdown tables section by section. Two table formats:
    - "Already Validated": | Model | Params | Arch | Quant | Size | T1 | T2 | T3 | T4 | Speedup | Notes |
    - Wave tables: | # | Model | Params | Arch | Quant | ~Size | Location | DL | T1 | T2 | T3 | T4 | PPL T-MAC | PPL Stock | Speedup |
    """
    with open(tracker_path, 'r') as f:
        lines = f.readlines()

    imported = 0

    def parse_params(p):
        """Parse '8B' or '57B/14B' → (total, active)."""
        if '/' in p:
            parts = p.replace('B', '').split('/')
            return float(parts[0]), float(parts[1])
        val = float(p.replace('B', ''))
        return val, val

    def parse_ppl(s):
        """Parse PPL value, returning None for '—' or 'NaN...'."""
        s = s.strip()
        if s in ('—', '-', '') or s.startswith('NaN'):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def parse_experts(arch):
        """Extract expert count from arch string like 'MoE (64E)'."""
        m = re.search(r'(\d+)E\)', arch)
        n_experts = int(m.group(1)) if m else None
        # Active experts: typically 2 for MoE, but varies
        return n_experts

    def upsert_model(model_name, params_b, active_params_b, arch, quant,
                     size_gb, t1, t2, t3, t4, speedup_str,
                     ppl_tmac=None, ppl_stock=None):
        """Insert or update a model record."""
        ppl_delta = None
        if ppl_tmac is not None and ppl_stock is not None:
            ppl_delta = round(abs(ppl_tmac - ppl_stock), 4)

        # Construct full model identifier
        model_key = f"{model_name}-{quant}" if quant not in model_name else model_name

        n_experts = parse_experts(arch)
        arch_clean = re.sub(r'\s*\(\d+E\)', '', arch).strip()

        conn.execute("""
            INSERT INTO models
                (model, base_model, quant_type, params_b, active_params_b,
                 arch_family, n_experts, size_gb,
                 tier1_status, tier2_status, tier3_status, tier4_status,
                 ppl_tmac, ppl_stock, ppl_delta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model) DO UPDATE SET
                base_model = excluded.base_model,
                quant_type = COALESCE(excluded.quant_type, models.quant_type),
                params_b = COALESCE(excluded.params_b, models.params_b),
                active_params_b = COALESCE(excluded.active_params_b, models.active_params_b),
                arch_family = COALESCE(excluded.arch_family, models.arch_family),
                n_experts = COALESCE(excluded.n_experts, models.n_experts),
                size_gb = COALESCE(excluded.size_gb, models.size_gb),
                tier1_status = COALESCE(excluded.tier1_status, models.tier1_status),
                tier2_status = COALESCE(excluded.tier2_status, models.tier2_status),
                tier3_status = COALESCE(excluded.tier3_status, models.tier3_status),
                tier4_status = COALESCE(excluded.tier4_status, models.tier4_status),
                ppl_tmac = COALESCE(excluded.ppl_tmac, models.ppl_tmac),
                ppl_stock = COALESCE(excluded.ppl_stock, models.ppl_stock),
                ppl_delta = COALESCE(excluded.ppl_delta, models.ppl_delta)
        """, (
            model_key, model_name, quant, params_b, active_params_b,
            arch_clean, n_experts, size_gb,
            t1 if t1 != '-' else None,
            t2 if t2 != '-' else None,
            t3 if t3 != '-' else None,
            t4 if t4 != '-' else None,
            ppl_tmac, ppl_stock, ppl_delta,
        ))
        return model_key

    # Detect table format from header row and parse accordingly
    # Two formats:
    #   "Already Validated": | Model | Params | Arch | Quant | Size | T1 | T2 | T3 | T4 | Speedup | Notes |
    #   Wave tables:         | # | Model | Params | Arch | Quant | ~Size | Location | DL | T1 | T2 | T3 | T4 | PPL T-MAC | PPL Stock | Speedup |
    current_format = None  # 'already' or 'wave'

    for line in lines:
        line = line.strip()

        # Detect table format from header rows
        if '| Model |' in line and '| Params |' in line:
            if '| # |' in line or '| Location |' in line:
                current_format = 'wave'
            else:
                current_format = 'already'
            continue

        # Skip separator rows
        if re.match(r'\|[\s\-:]+\|', line):
            continue
        # Non-table lines reset format detection
        if not line.startswith('|'):
            current_format = None
            continue

        # Parse table cells — strip leading/trailing empty cells from pipes
        cells = [c.strip() for c in line.split('|')]
        # Remove leading/trailing empty strings from |...|
        while cells and cells[0] == '':
            cells.pop(0)
        while cells and cells[-1] == '':
            cells.pop()
        if len(cells) < 5:
            continue

        try:
            if current_format == 'already':
                # | Model | Params | Arch | Quant | Size | T1 | T2 | T3 | T4 | Speedup | Notes |
                # Skip header/separator detection
                if cells[0] in ('Model', '-------', ''):
                    continue
                model_name = cells[0]
                params_str = cells[1]
                if not re.match(r'[\d.]+B', params_str):
                    continue
                params_b, active_params_b = parse_params(params_str)
                arch = cells[2]
                quant = cells[3]
                size_match = re.search(r'([\d.]+)', cells[4])
                size_gb = float(size_match.group(1)) if size_match else None
                t1 = cells[5] if len(cells) > 5 else '-'
                t2 = cells[6] if len(cells) > 6 else '-'
                t3 = cells[7] if len(cells) > 7 else '-'
                t4 = cells[8] if len(cells) > 8 else '-'
                speedup_str = cells[9] if len(cells) > 9 else ''

                upsert_model(model_name, params_b, active_params_b, arch, quant,
                             size_gb, t1, t2, t3, t4, speedup_str)
                imported += 1

            elif current_format == 'wave':
                # | # | Model | Params | Arch | Quant | ~Size | Location | DL | T1 | T2 | T3 | T4 | PPL T-MAC | PPL Stock | Speedup |
                # First cell is row number
                if not re.match(r'\d+', cells[0]):
                    continue
                model_name = cells[1]
                params_str = cells[2]
                if not re.match(r'[\d.]+B', params_str):
                    continue
                params_b, active_params_b = parse_params(params_str)
                arch = cells[3]
                quant = cells[4]
                size_match = re.search(r'([\d.]+)', cells[5])
                size_gb = float(size_match.group(1)) if size_match else None
                # cells[6] = Location, cells[7] = DL
                t1 = cells[8] if len(cells) > 8 else '-'
                t2 = cells[9] if len(cells) > 9 else '-'
                t3 = cells[10] if len(cells) > 10 else '-'
                t4 = cells[11] if len(cells) > 11 else '-'
                ppl_tmac_str = cells[12] if len(cells) > 12 else '—'
                ppl_stock_str = cells[13] if len(cells) > 13 else '—'
                speedup_str = cells[14] if len(cells) > 14 else ''

                ppl_tmac = parse_ppl(ppl_tmac_str)
                ppl_stock = parse_ppl(ppl_stock_str)

                upsert_model(model_name, params_b, active_params_b, arch, quant,
                             size_gb, t1, t2, t3, t4, speedup_str,
                             ppl_tmac, ppl_stock)
                imported += 1

        except (ValueError, IndexError):
            continue  # Skip malformed rows

    conn.commit()
    return imported


# ── Enrich Models ──────────────────────────────────────────────────────

def enrich_models(conn, config_path):
    """Enrich models table from a JSON metadata file.

    JSON format: { "model-key": { "bpw": 4.83, "n_active": 2, ... }, ... }
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    updated = 0
    for model_key, fields in config.items():
        # Build dynamic UPDATE
        allowed = {
            'base_model', 'quant_type', 'params_b', 'active_params_b',
            'arch_family', 'n_experts', 'n_active', 'size_gb', 'bpw',
        }
        sets = []
        vals = []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k} = ?")
                vals.append(v)

        if not sets:
            continue

        vals.append(model_key)
        conn.execute(
            f"UPDATE models SET {', '.join(sets)} WHERE model = ?",
            vals
        )
        # Also try matching by base_model-quant pattern (different WHERE)
        conn.execute(
            f"UPDATE models SET {', '.join(sets)} WHERE model LIKE ? AND model != ?",
            vals[:-1] + [f"%{model_key}%", model_key]
        )
        updated += 1

    conn.commit()
    return updated


# ── Status ─────────────────────────────────────────────────────────────

def print_status(conn, model_filter=None):
    """Print database status summary."""
    counts = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM benchmark_sessions) as sessions,
            (SELECT COUNT(*) FROM measurements) as measurements,
            (SELECT COUNT(*) FROM computed_stats) as stats,
            (SELECT COUNT(*) FROM models) as models,
            (SELECT COUNT(*) FROM system_telemetry) as telemetry
    """).fetchone()

    print(f"\n{'='*60}")
    print(f"  Benchmark Database Status")
    print(f"{'='*60}")
    print(f"  Sessions:     {counts['sessions']}")
    print(f"  Measurements: {counts['measurements']}")
    print(f"  Stats:        {counts['stats']}")
    print(f"  Models:       {counts['models']}")
    print(f"  Telemetry:    {counts['telemetry']}")
    print()

    # List sessions
    rows = conn.execute("""
        SELECT bs.id, bs.source_file, bs.session_date, bs.gpu_mode,
               bs.n_effective, bs.methodology,
               COUNT(m.id) as n_meas,
               COUNT(DISTINCT m.model) as n_models
        FROM benchmark_sessions bs
        LEFT JOIN measurements m ON m.session_id = bs.id AND m.is_warmup = 0
        GROUP BY bs.id
        ORDER BY bs.id
    """).fetchall()

    if rows:
        print("  Sessions:")
        for r in rows:
            mode = f" [{r['gpu_mode']}]" if r['gpu_mode'] else ""
            print(f"    #{r['id']:2d}  {r['source_file']:45s}  "
                  f"N={r['n_effective'] or '?':>2}{mode}  "
                  f"{r['n_meas']} meas, {r['n_models']} model(s)")
        print()

    # Model-specific detail
    if model_filter:
        stat_rows = conn.execute("""
            SELECT cs.*, bs.source_file, bs.gpu_mode
            FROM computed_stats cs
            JOIN benchmark_sessions bs ON cs.session_id = bs.id
            WHERE cs.model LIKE ?
            ORDER BY cs.computed_at DESC
        """, (f'%{model_filter}%',)).fetchall()

        if stat_rows:
            print(f"  Stats for '{model_filter}':")
            for r in stat_rows:
                sig = "*" if r['is_significant'] else " "
                mode = f" [{r['gpu_mode']}]" if r['gpu_mode'] else ""
                print(f"    {r['model']:45s}  "
                      f"T-MAC: {r['tmac_mean']:8.2f} ± {r['tmac_sd']:5.2f}  "
                      f"Stock: {r['stock_mean']:8.2f} ± {r['stock_sd']:5.2f}  "
                      f"Δ={r['speedup_pct']:+.1f}% [{r['speedup_ci95_lo']:+.1f}, "
                      f"{r['speedup_ci95_hi']:+.1f}]{sig}  "
                      f"p={r['p_value']:.4f}  N={r['n_pairs']}{mode}")
            print()
        else:
            print(f"  No stats found matching '{model_filter}'")
    else:
        # Show top speedups
        top = conn.execute("""
            SELECT model, speedup_pct, speedup_ci95_lo, speedup_ci95_hi,
                   n_pairs, p_value, is_significant
            FROM computed_stats
            ORDER BY speedup_pct DESC
            LIMIT 10
        """).fetchall()

        if top:
            print("  Top speedups:")
            for r in top:
                sig = "*" if r['is_significant'] else " "
                print(f"    {r['model']:45s}  "
                      f"{r['speedup_pct']:+6.1f}% "
                      f"[{r['speedup_ci95_lo']:+.1f}, {r['speedup_ci95_hi']:+.1f}]{sig}  "
                      f"N={r['n_pairs']}  p={r['p_value']:.4f}")
            print()

    print(f"{'='*60}")


# ── Export ─────────────────────────────────────────────────────────────

def export_chart_data(conn, chart_name):
    """Export chart data as JSON to stdout."""
    queries = {
        'q4km': lambda: bench_db.query_chart_q4km(conn),
        'iq': lambda: bench_db.query_chart_iq(conn),
        'multigpu': lambda: bench_db.query_chart_multigpu(conn),
        'absolute': lambda: bench_db.query_chart_absolute(conn),
    }

    if chart_name not in queries:
        print(f"Unknown chart: {chart_name}. Available: {', '.join(queries.keys())}")
        return

    data = queries[chart_name]()
    print(json.dumps(data, indent=2))


def export_table(conn, table_name, fmt='md'):
    """Export a summary table."""
    if table_name == 'stats':
        rows = conn.execute("""
            SELECT cs.model, cs.metric, cs.n_pairs,
                   cs.tmac_mean, cs.stock_mean,
                   cs.speedup_pct, cs.speedup_ci95_lo, cs.speedup_ci95_hi,
                   cs.p_value, cs.is_significant,
                   bs.gpu_mode
            FROM computed_stats cs
            JOIN benchmark_sessions bs ON cs.session_id = bs.id
            ORDER BY cs.speedup_pct DESC
        """).fetchall()

        if fmt == 'md':
            print("| Model | N | T-MAC | Stock | Speedup | 95% CI | p-value | GPU |")
            print("|-------|--:|------:|------:|--------:|-------:|--------:|-----|")
            for r in rows:
                sig = "**" if r['is_significant'] else ""
                mode = r['gpu_mode'] or 'single'
                print(f"| {r['model']} | {r['n_pairs']} | "
                      f"{r['tmac_mean']:.2f} | {r['stock_mean']:.2f} | "
                      f"{sig}{r['speedup_pct']:+.1f}%{sig} | "
                      f"[{r['speedup_ci95_lo']:+.1f}, {r['speedup_ci95_hi']:+.1f}] | "
                      f"{r['p_value']:.4f} | {mode} |")
        elif fmt == 'csv':
            w = csv.writer(sys.stdout)
            w.writerow(['model', 'n_pairs', 'tmac_mean', 'stock_mean',
                        'speedup_pct', 'ci95_lo', 'ci95_hi', 'p_value',
                        'significant', 'gpu_mode'])
            for r in rows:
                w.writerow([r['model'], r['n_pairs'], r['tmac_mean'],
                            r['stock_mean'], r['speedup_pct'],
                            r['speedup_ci95_lo'], r['speedup_ci95_hi'],
                            r['p_value'], r['is_significant'],
                            r['gpu_mode'] or 'single'])

    elif table_name == 'models':
        rows = conn.execute("""
            SELECT * FROM models ORDER BY params_b DESC, model
        """).fetchall()

        if fmt == 'md':
            print("| Model | Params | Arch | Quant | Size | T1 | T2 | T3 | T4 | PPL Δ |")
            print("|-------|-------:|------|-------|-----:|:--:|:--:|:--:|:--:|------:|")
            for r in rows:
                ppl_d = f"{r['ppl_delta']:.3f}" if r['ppl_delta'] is not None else '—'
                print(f"| {r['model']} | {r['params_b'] or '?'}B | "
                      f"{r['arch_family'] or '?'} | {r['quant_type'] or '?'} | "
                      f"{r['size_gb'] or '?'} GB | "
                      f"{r['tier1_status'] or '-'} | {r['tier2_status'] or '-'} | "
                      f"{r['tier3_status'] or '-'} | {r['tier4_status'] or '-'} | "
                      f"{ppl_d} |")
    else:
        print(f"Unknown table: {table_name}. Available: stats, models")


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark import tool for kuzco.cpp',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--db', default=DEFAULT_DB,
                        help=f'Database path (default: {DEFAULT_DB})')

    sub = parser.add_subparsers(dest='command', help='Command')

    # init
    p_init = sub.add_parser('init', help='Create database with schema')
    p_init.add_argument('--force', action='store_true',
                        help='Drop and recreate all tables')

    # import
    p_import = sub.add_parser('import', help='Import a CSV file')
    p_import.add_argument('csv_file', help='Path to CSV file')
    p_import.add_argument('--telemetry', action='store_true',
                          help='Capture system telemetry at import time')
    p_import.add_argument('--model', help='Override model name (for Format C)')
    p_import.add_argument('--warmup', type=int, default=0,
                          help='Number of warmup runs to exclude')

    # import-dir
    p_importdir = sub.add_parser('import-dir', help='Import all CSVs in directory')
    p_importdir.add_argument('directory', help='Directory containing CSV files')
    p_importdir.add_argument('--telemetry', action='store_true')

    # import-zoo
    p_zoo = sub.add_parser('import-zoo', help='Import model metadata from tracker')
    p_zoo.add_argument('tracker', help='Path to model-zoo-tracker.md')

    # recompute
    p_recompute = sub.add_parser('recompute', help='Recompute stats from measurements')
    p_recompute.add_argument('--session-id', type=int,
                             help='Only recompute for this session')

    # enrich
    p_enrich = sub.add_parser('enrich', help='Enrich models from JSON config')
    p_enrich.add_argument('--config', required=True, help='Path to JSON file')

    # status
    p_status = sub.add_parser('status', help='Show database status')
    p_status.add_argument('--model', help='Filter by model name')

    # export-chart-data
    p_chart = sub.add_parser('export-chart-data', help='Export chart data as JSON')
    p_chart.add_argument('chart', help='Chart name: q4km, iq, multigpu, absolute')

    # export-table
    p_table = sub.add_parser('export-table', help='Export summary table')
    p_table.add_argument('table', help='Table name: stats, models')
    p_table.add_argument('--format', choices=['md', 'csv'], default='md')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    db_path = os.path.abspath(args.db)

    # ── init ────────────────────────────────────────────────────────
    if args.command == 'init':
        if args.force and os.path.exists(db_path):
            os.remove(db_path)
            print(f"Removed existing DB: {db_path}")
        conn = bench_db.connect(db_path)
        bench_db.create_schema(conn)
        print(f"Database initialized: {db_path}")
        print(f"Schema version: {bench_db.SCHEMA_VERSION}")
        conn.close()
        return

    # All other commands need existing DB
    if not os.path.exists(db_path) and args.command != 'init':
        print(f"Database not found: {db_path}")
        print(f"Run: bench-import.py init")
        sys.exit(1)

    conn = bench_db.connect(db_path)

    # ── import ──────────────────────────────────────────────────────
    if args.command == 'import':
        filepath = os.path.abspath(args.csv_file)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            sys.exit(1)
        sid = import_csv(conn, filepath, telemetry=args.telemetry,
                         model_override=args.model, warmup=args.warmup)
        if sid:
            n = recompute_stats(conn, session_id=sid)
            print(f"  Computed {n} stat group(s)")

    # ── import-dir ──────────────────────────────────────────────────
    elif args.command == 'import-dir':
        dirpath = os.path.abspath(args.directory)
        if not os.path.isdir(dirpath):
            print(f"Not a directory: {dirpath}")
            sys.exit(1)

        csv_files = sorted(
            f for f in os.listdir(dirpath) if f.endswith('.csv')
        )
        if not csv_files:
            print(f"No CSV files found in {dirpath}")
            return

        print(f"Found {len(csv_files)} CSV file(s) in {dirpath}")
        imported_sessions = []
        for f in csv_files:
            filepath = os.path.join(dirpath, f)
            sid = import_csv(conn, filepath, telemetry=args.telemetry)
            if sid:
                imported_sessions.append(sid)

        # Recompute all stats
        if imported_sessions:
            total = 0
            for sid in imported_sessions:
                total += recompute_stats(conn, session_id=sid)
            print(f"\nComputed {total} stat group(s) across "
                  f"{len(imported_sessions)} session(s)")

    # ── import-zoo ──────────────────────────────────────────────────
    elif args.command == 'import-zoo':
        filepath = os.path.abspath(args.tracker)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            sys.exit(1)
        n = import_zoo(conn, filepath)
        print(f"Imported {n} model(s) from zoo tracker")

    # ── recompute ───────────────────────────────────────────────────
    elif args.command == 'recompute':
        n = recompute_stats(conn, session_id=args.session_id)
        scope = f"session #{args.session_id}" if args.session_id else "all sessions"
        print(f"Recomputed {n} stat group(s) for {scope}")

    # ── enrich ──────────────────────────────────────────────────────
    elif args.command == 'enrich':
        filepath = os.path.abspath(args.config)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            sys.exit(1)
        n = enrich_models(conn, filepath)
        print(f"Enriched {n} model(s)")

    # ── status ──────────────────────────────────────────────────────
    elif args.command == 'status':
        print_status(conn, model_filter=args.model)

    # ── export-chart-data ───────────────────────────────────────────
    elif args.command == 'export-chart-data':
        export_chart_data(conn, args.chart)

    # ── export-table ────────────────────────────────────────────────
    elif args.command == 'export-table':
        export_table(conn, args.table, fmt=args.format)

    conn.close()


if __name__ == '__main__':
    main()
