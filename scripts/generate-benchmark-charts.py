#!/usr/bin/env python3
"""Generate benchmark charts for kuzco.cpp README.

Usage:
    python3 scripts/generate-benchmark-charts.py              # legacy (hardcoded)
    python3 scripts/generate-benchmark-charts.py --legacy      # explicit legacy
    python3 scripts/generate-benchmark-charts.py --db PATH     # from SQLite DB

Output: docs/tmac/chart-*.png
"""

import argparse
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import sys

# Add scripts dir for bench_db import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_db

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'tmac')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color palette (AMD red accent, neutral grays) ─────────────────────
AMD_RED = '#ED1C24'
AMD_RED_LIGHT = '#FF6B6B'
AMD_DARK = '#1A1A2E'
GRAY_BG = '#F8F9FA'
GRAY_GRID = '#E0E0E0'
GRAY_BAR = '#B0B0B0'
ACCENT_BLUE = '#4A90D9'
ACCENT_TEAL = '#2ECC71'


def style_ax(ax, title, xlabel=None, ylabel=None):
    """Apply consistent clean styling."""
    ax.set_facecolor(GRAY_BG)
    ax.set_title(title, fontsize=14, fontweight='bold', color=AMD_DARK, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color='#555')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color='#555')
    ax.tick_params(colors='#555', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRAY_GRID)
    ax.spines['bottom'].set_color(GRAY_GRID)
    ax.grid(axis='x', linestyle='-', alpha=0.4, color=GRAY_GRID)


# ═══════════════════════════════════════════════════════════════════════
# Chart 1: Q4_K_M Speedup (horizontal bar chart)
# ═══════════════════════════════════════════════════════════════════════
def chart_q4km(data=None, output_suffix=''):
    if data is None:
        # Legacy hardcoded data
        models = [
            'QwQ-32B',
            'Qwen3.5-9B',
            'Qwen3.5-35B\n(MoE)',
            'Codestral 22B',
            'OLMoE-1B-7B\n(MoE)',
            'GLM-4.7-Flash\n(MLA+MoE)',
            'DeepSeek-V2-Lite\n(MoE+MLA)',
            'Llama 3.2 1B',
        ]
        stock =  [29.88, 69.8, 75.0, 40.03, 324.77, 87.36, 155.17, 373.43]
        tmac =   [33.92, 77.6, 83.7, 45.66, 372.87, 100.67, 179.78, 449.38]
        speedup = [100*(t-s)/s for t, s in zip(tmac, stock)]
    else:
        # DB-driven data
        models = [_format_model_label(d) for d in data]
        stock = [d['stock_mean'] for d in data]
        tmac = [d['tmac_mean'] for d in data]
        speedup = [d['speedup_pct'] for d in data]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('white')

    y = np.arange(len(models))
    bars = ax.barh(y, speedup, height=0.6, color=AMD_RED, edgecolor='white', linewidth=0.5)

    # Value labels
    for bar, sp in zip(bars, speedup):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'+{sp:.1f}%', va='center', ha='left', fontsize=10, fontweight='bold', color=AMD_DARK)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlim(0, max(speedup) * 1.25)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    style_ax(ax, 'T-MAC Speedup — Q4_K_M Quantization', xlabel='Speedup over stock llama.cpp')

    ax.text(0.98, 0.02, 'N=10 paired interleaved · single RX 7900 XTX · tg128',
            transform=ax.transAxes, fontsize=7.5, color='#999', ha='right', va='bottom')

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f'chart-q4km-speedup{output_suffix}.png')
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {path}')


# ═══════════════════════════════════════════════════════════════════════
# Chart 2: IQ Types Speedup (grouped — shows bpw on secondary axis)
# ═══════════════════════════════════════════════════════════════════════
def chart_iq(data=None, output_suffix=''):
    if data is None:
        # Legacy hardcoded data
        labels = [
            'Llama 1B · IQ1_M\n1.75 bpw',
            'Llama 1B · IQ2_XS\n2.31 bpw',
            'OLMoE · IQ2_XXS\n2.06 bpw (MoE)',
            'Llama 70B · IQ2_XXS\n2.06 bpw',
            'Llama 1B · IQ2_XXS\n2.06 bpw',
            'DBRX · IQ2_XXS\n2.06 bpw (MoE)',
            'Llama 1B · IQ3_S\n3.44 bpw',
            'Llama 1B · IQ3_XXS\n3.06 bpw',
            'Jamba · IQ3_XXS\n3.06 bpw (MoE)',
            'Qwen2-57B · IQ3_XXS\n3.06 bpw (MoE)',
        ]
        speedup = [11.9, 17.0, 18.3, 25.8, 24.4, 22.0, 34.4, 36.9, 47.2, 54.5]
        is_moe = [False, False, True, False, False, True, False, False, True, True]
    else:
        labels = [_format_iq_label(d) for d in data]
        speedup = [d['speedup_pct'] for d in data]
        is_moe = [_is_moe(d) for d in data]

    # Color: MoE models in teal, dense in red gradient
    colors = [ACCENT_TEAL if m else plt.cm.Reds(0.3 + 0.6 * s / max(speedup))
              for s, m in zip(speedup, is_moe)]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')

    y = np.arange(len(labels))
    bars = ax.barh(y, speedup, height=0.6, color=colors, edgecolor='white', linewidth=0.5)

    for bar, sp in zip(bars, speedup):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'+{sp:.1f}%', va='center', ha='left', fontsize=9, fontweight='bold', color=AMD_DARK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, max(speedup) * 1.2)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    style_ax(ax, 'T-MAC Speedup — IQ Quantization Types', xlabel='Speedup over stock llama.cpp')

    # Legend for MoE vs Dense
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=AMD_RED, label='Dense models'),
                       Patch(facecolor=ACCENT_TEAL, label='MoE models')],
              loc='lower right', framealpha=0.9, fontsize=8.5)

    ax.text(0.98, 0.02, 'N≥5 paired interleaved · single RX 7900 XTX · tg128',
            transform=ax.transAxes, fontsize=7.5, color='#999', ha='right', va='bottom')

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f'chart-iq-speedup{output_suffix}.png')
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {path}')


# ═══════════════════════════════════════════════════════════════════════
# Chart 3: Absolute throughput comparison (stock vs T-MAC, Q4_K_M)
# ═══════════════════════════════════════════════════════════════════════
def chart_absolute(data=None, output_suffix=''):
    if data is None:
        models = ['Llama 1B', 'OLMoE\n(MoE)', 'DSV2-Lite\n(MoE)', 'GLM Flash\n(MLA+MoE)', 'Qwen3.5-9B', 'Qwen3.5-35B\n(MoE)', 'Codestral\n22B', 'QwQ\n32B']
        stock  = [373.43, 324.77, 155.17, 87.36, 69.8, 75.0, 40.03, 29.88]
        tmac   = [449.38, 372.87, 179.78, 100.67, 77.6, 83.7, 45.66, 33.92]
    else:
        models = [_format_model_label(d) for d in data]
        stock = [d['stock_mean'] for d in data]
        tmac = [d['tmac_mean'] for d in data]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('white')

    x = np.arange(len(models))
    w = 0.35

    bars_stock = ax.bar(x - w/2, stock, w, label='stock llama.cpp', color=GRAY_BAR, edgecolor='white', linewidth=0.5)
    bars_tmac = ax.bar(x + w/2, tmac, w, label='kuzco.cpp (T-MAC)', color=AMD_RED, edgecolor='white', linewidth=0.5)

    # Speedup labels above T-MAC bars
    for i, (s, t) in enumerate(zip(stock, tmac)):
        sp = 100 * (t - s) / s
        ax.text(x[i] + w/2, t + 5, f'+{sp:.0f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=AMD_RED)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, max(tmac) * 1.18)

    style_ax(ax, 'Token Generation Throughput — Q4_K_M on RX 7900 XTX', ylabel='Tokens per second (tg128)')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.text(0.98, 0.01, 'N=10 paired interleaved · higher is better',
             fontsize=7.5, color='#999', ha='right', va='bottom')
    path = os.path.join(OUT_DIR, f'chart-throughput-q4km{output_suffix}.png')
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {path}')


# ═══════════════════════════════════════════════════════════════════════
# Chart 4: Multi-GPU absolute throughput (stock vs T-MAC, dual 7900 XTX)
# ═══════════════════════════════════════════════════════════════════════
def chart_multigpu(data=None, output_suffix=''):
    if data is None:
        models = [
            'Mixtral 8x7B\nIQ3_S',
            'DBRX\nIQ2_XXS',
            '70B IQ2_XXS',
            'Qwen2-57B\nQ4_K_M',
            'Llama 4 Scout\nIQ2_XXS-UD',
            '70B Q4_0',
        ]
        stock = [55.47, 23.09, 19.26, 53.70, 39.84, 20.83]
        tmac  = [80.67, 28.16, 22.90, 60.71, 44.62, 22.19]
    else:
        models = [_format_model_label(d) for d in data]
        stock = [d['stock_mean'] for d in data]
        tmac = [d['tmac_mean'] for d in data]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('white')

    x = np.arange(len(models))
    w = 0.35

    ax.bar(x - w/2, stock, w, label='stock llama.cpp', color=GRAY_BAR, edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, tmac, w, label='kuzco.cpp (T-MAC)', color=AMD_RED, edgecolor='white', linewidth=0.5)

    # Speedup labels above T-MAC bars
    for i, (s, t) in enumerate(zip(stock, tmac)):
        sp = 100 * (t - s) / s
        ax.text(x[i] + w/2, t + 1.2, f'+{sp:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=AMD_RED)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, max(tmac) * 1.18)

    style_ax(ax, 'Multi-GPU Throughput — Dual RX 7900 XTX', ylabel='Tokens per second (tg128)')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.text(0.98, 0.01, 'N≥10 paired · dual 7900 XTX · higher is better',
             fontsize=7.5, color='#999', ha='right', va='bottom')
    path = os.path.join(OUT_DIR, f'chart-multigpu-throughput{output_suffix}.png')
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {path}')


# ── DB-mode helpers ────────────────────────────────────────────────────

def _format_model_label(d):
    """Format a DB row dict into a chart-friendly label."""
    model = d.get('base_model') or d.get('model', '?')
    arch = d.get('arch_family') or ''  # may be None from SQL LEFT JOIN
    suffix = f'\n({arch})' if arch and ('MoE' in arch or 'MLA' in arch) else ''
    return f'{model}{suffix}'


def _format_iq_label(d):
    """Format IQ chart label with bpw annotation."""
    model = d.get('base_model') or d.get('model', '?')
    qt = bench_db.extract_quant_type(d.get('model', ''))
    bpw = d.get('bpw', '')
    bpw_str = f'{bpw} bpw' if bpw else ''
    moe_str = ' (MoE)' if _is_moe(d) else ''
    return f'{model} · {qt}\n{bpw_str}{moe_str}'.strip()


def _is_moe(d):
    """Check if model is MoE from DB row."""
    arch = d.get('arch_family', '') or ''
    return 'MoE' in arch


def load_db_data(db_path):
    """Load all chart data from DB. Returns dict of chart_name -> data."""
    conn = bench_db.connect(db_path)
    data = {
        'q4km': bench_db.query_chart_q4km(conn),
        'iq': bench_db.query_chart_iq(conn),
        'multigpu': bench_db.query_chart_multigpu(conn),
        'absolute': bench_db.query_chart_absolute(conn),
    }
    conn.close()
    return data


# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate benchmark charts')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--db', metavar='PATH', help='Read data from SQLite DB')
    mode.add_argument('--legacy', action='store_true',
                      help='Use hardcoded data (default if no --db)')
    parser.add_argument('--output-suffix', default='',
                        help='Suffix for output filenames (e.g., "-legacy")')
    args = parser.parse_args()

    suffix = args.output_suffix

    print('Generating benchmark charts...')

    if args.db:
        if not os.path.exists(args.db):
            print(f'DB not found: {args.db}')
            sys.exit(1)
        data = load_db_data(args.db)
        charts = [
            ('q4km', chart_q4km, 'Q4_K_M'),
            ('iq', chart_iq, 'IQ'),
            ('absolute', chart_absolute, 'absolute'),
            ('multigpu', chart_multigpu, 'multi-GPU'),
        ]
        for key, fn, label in charts:
            if data[key]:
                fn(data[key], output_suffix=suffix)
            else:
                print(f'  SKIP chart-{key} (no {label} data in DB)')
    else:
        # Legacy mode (hardcoded data)
        chart_q4km(output_suffix=suffix)
        chart_iq(output_suffix=suffix)
        chart_absolute(output_suffix=suffix)
        chart_multigpu(output_suffix=suffix)

    print('Done.')
