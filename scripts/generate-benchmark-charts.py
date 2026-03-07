#!/usr/bin/env python3
"""Generate benchmark charts for kuzco.cpp README.

Usage: python3 scripts/generate-benchmark-charts.py
Output: docs/tmac/chart-*.png
"""

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

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
def chart_q4km():
    models = [
        'QwQ-32B',
        'Codestral 22B',
        'Ministral 14B',
        'OLMoE-1B-7B\n(MoE)',
        'GLM-4.7-Flash\n(MLA+MoE)',
        'Llama 3.2 1B',
    ]
    stock =  [29.88, 40.03, 64.04, 324.77, 87.36, 373.43]
    tmac =   [33.92, 45.66, 73.04, 372.87, 100.67, 449.38]
    speedup = [100*(t-s)/s for t, s in zip(tmac, stock)]

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
    path = os.path.join(OUT_DIR, 'chart-q4km-speedup.png')
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {path}')


# ═══════════════════════════════════════════════════════════════════════
# Chart 2: IQ Types Speedup (grouped — shows bpw on secondary axis)
# ═══════════════════════════════════════════════════════════════════════
def chart_iq():
    types = [
        'IQ1_M\n1.75 bpw',
        'IQ2_XS\n2.31 bpw',
        'IQ2_XXS\n2.06 bpw',
        'IQ3_S\n3.44 bpw',
        'IQ3_XXS\n3.06 bpw',
    ]
    # All Llama 3.2 1B, N=10
    speedup = [11.9, 17.0, 24.4, 34.4, 36.9]

    # Color gradient: darker red = more speedup
    colors = [plt.cm.Reds(0.3 + 0.6 * s / max(speedup)) for s in speedup]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('white')

    y = np.arange(len(types))
    bars = ax.barh(y, speedup, height=0.6, color=colors, edgecolor='white', linewidth=0.5)

    for bar, sp in zip(bars, speedup):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'+{sp:.1f}%', va='center', ha='left', fontsize=10, fontweight='bold', color=AMD_DARK)

    ax.set_yticks(y)
    ax.set_yticklabels(types, fontsize=10)
    ax.set_xlim(0, max(speedup) * 1.25)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    style_ax(ax, 'T-MAC Speedup — IQ Quantization Types (Llama 3.2 1B)', xlabel='Speedup over stock llama.cpp')

    ax.text(0.98, 0.02, 'N=10 paired interleaved · single RX 7900 XTX · tg128',
            transform=ax.transAxes, fontsize=7.5, color='#999', ha='right', va='bottom')

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'chart-iq-speedup.png')
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {path}')


# ═══════════════════════════════════════════════════════════════════════
# Chart 3: Absolute throughput comparison (stock vs T-MAC, Q4_K_M)
# ═══════════════════════════════════════════════════════════════════════
def chart_absolute():
    models = ['Llama 1B', 'OLMoE\n(MoE)', 'GLM Flash\n(MLA+MoE)', 'Ministral\n14B', 'Codestral\n22B', 'QwQ\n32B']
    stock  = [373.43, 324.77, 87.36, 64.04, 40.03, 29.88]
    tmac   = [449.38, 372.87, 100.67, 73.04, 45.66, 33.92]

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
    path = os.path.join(OUT_DIR, 'chart-throughput-q4km.png')
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {path}')


# ═══════════════════════════════════════════════════════════════════════
# Chart 4: Multi-GPU absolute throughput (stock vs T-MAC, dual 7900 XTX)
# ═══════════════════════════════════════════════════════════════════════
def chart_multigpu():
    models = [
        'Mixtral 8x7B\nIQ3_S',
        'Llama 4 Scout\nIQ2_XXS-UD',
        '70B IQ2_XXS',
        '70B Q4_0',
    ]
    stock = [55.47, 39.84, 19.26, 20.83]
    tmac  = [80.67, 44.62, 22.90, 22.19]

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
    path = os.path.join(OUT_DIR, 'chart-multigpu-throughput.png')
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {path}')


# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating benchmark charts...')
    chart_q4km()
    chart_iq()
    chart_absolute()
    chart_multigpu()
    print('Done.')
