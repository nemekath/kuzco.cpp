#!/bin/bash
# Scout Q4_K_M N=10 interleaved benchmark (rerun for statistical validation)
# 12 runs (2 warmup + 10 measured), interleaved T-MAC / Stock
# Dual GPU with CPU offload (-ngl 36), --mmap 1 for large model
set -euo pipefail

MODEL="/home/benjamin/llama-tmac-real/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf"
BENCH="/home/benjamin/llama-tmac-master/build/bin/llama-bench"
OUTCSV="/home/benjamin/llama-tmac-master/data/benchmarks/scout-q4km-n10-rerun.csv"
NGL=36
TOTAL_RUNS=12  # 2 warmup + 10 measured

# Extract avg_ts from llama-bench CSV (handles quoted commas in gpu_info)
extract_ts() {
    python3 -c "
import csv, sys
reader = csv.reader(sys.stdin)
next(reader)  # skip header
for row in reader:
    print(row[-2])  # avg_ts is second-to-last field
"
}

echo "# model=Scout-Q4_K_M-rerun,date=$(date -Iseconds),gpu_mode=dual,ngl=${NGL}" > "$OUTCSV"
echo "run,type,avg_ts,timestamp" >> "$OUTCSV"

echo "=== Scout Q4_K_M N=10 Interleaved Benchmark ==="
echo "Model: $MODEL"
echo "Config: HIP_VISIBLE_DEVICES=0,1 -ngl $NGL --mmap 1"
echo "Runs: $TOTAL_RUNS (first 2 = warmup)"
echo "Output: $OUTCSV"
echo ""

for run in $(seq 1 $TOTAL_RUNS); do
    echo "--- Run $run/$TOTAL_RUNS (T-MAC) ---"
    ts_tmac=$(HIP_VISIBLE_DEVICES=0,1 "$BENCH" \
        -m "$MODEL" -p 0 -n 128 -ngl $NGL --mmap 1 -r 1 \
        -o csv 2>/dev/null | extract_ts)
    echo "  T-MAC: ${ts_tmac} t/s"
    echo "${run},tmac,${ts_tmac},$(date -Iseconds)" >> "$OUTCSV"

    echo "--- Run $run/$TOTAL_RUNS (Stock) ---"
    ts_stock=$(HIP_VISIBLE_DEVICES=0,1 GGML_HIP_NO_TMAC=1 "$BENCH" \
        -m "$MODEL" -p 0 -n 128 -ngl $NGL --mmap 1 -r 1 \
        -o csv 2>/dev/null | extract_ts)
    echo "  Stock: ${ts_stock} t/s"
    echo "${run},stock,${ts_stock},$(date -Iseconds)" >> "$OUTCSV"

    echo ""
done

echo "=== Benchmark complete ==="
echo "Results saved to: $OUTCSV"
echo ""
echo "Discard runs 1-2 (warmup). Compute paired t-test on runs 3-12."
