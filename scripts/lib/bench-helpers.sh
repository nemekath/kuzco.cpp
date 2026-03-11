#!/usr/bin/env bash
# Shared helpers for T-MAC benchmark scripts.
# Source this file, don't execute it.
#
# Required: $BENCH must be set to the llama-bench binary path.
# Optional: $GPU_ARGS (extra args like --split-mode, -ngl override)

# Parse avg_ts from llama-bench CSV output (header + data line).
# Usage: parse_avg_ts "$raw_csv_output"
parse_avg_ts() {
    local avg_ts
    avg_ts=$(echo "$1" | awk -F',' '
        NR==1 { for(i=1;i<=NF;i++) if($i=="avg_ts") col=i }
        NR==2 && col { gsub(/"/, "", $col); print $col }
    ')
    if [[ -z "$avg_ts" ]]; then
        echo "ERROR: failed to parse avg_ts from llama-bench output" >&2
        return 1
    fi
    printf '%s' "$avg_ts"
}

# Run a single llama-bench invocation and extract avg_ts from CSV output.
# Usage: run_bench <model> <metric> [env_prefix]
#   metric: tg128, pp512, tg512, tg2048
#   env_prefix: e.g. "GGML_HIP_NO_TMAC=1" (optional)
run_bench() {
    local model="$1" metric="$2" env_prefix="${3:-}"
    local args="-m $model -r 1 -ngl 99 -o csv ${GPU_ARGS:-}"

    case "$metric" in
        pp512)   args="$args -p 512 -n 0" ;;
        tg128)   args="$args -p 0 -n 128" ;;
        tg512)   args="$args -p 0 -n 512" ;;
        tg2048)  args="$args -p 0 -n 2048" ;;
    esac

    local raw
    if [[ -n "$env_prefix" ]]; then
        raw=$(env $env_prefix $BENCH $args 2>/dev/null)
    else
        raw=$($BENCH $args 2>/dev/null)
    fi
    parse_avg_ts "$raw"
}

# Compute arithmetic mean of values passed as arguments.
compute_mean() {
    local arr=("$@")
    local n="${#arr[@]}"
    if (( n == 0 )); then printf "0.0000"; return; fi
    awk -v n="$n" 'BEGIN {s=0; for(i=1;i<ARGC;i++) s+=ARGV[i]; printf "%.4f", s/n}' "${arr[@]}"
}

# Compute sample standard deviation (Bessel-corrected, n-1).
compute_sd() {
    local arr=("$@")
    local n="${#arr[@]}"
    if (( n < 2 )); then printf "0.0000"; return; fi
    awk -v n="$n" 'BEGIN {
        s=0; for(i=1;i<ARGC;i++) s+=ARGV[i]; m=s/n;
        ss=0; for(i=1;i<ARGC;i++) ss+=(ARGV[i]-m)^2;
        printf "%.4f", sqrt(ss/(n-1))
    }' "${arr[@]}"
}

# Paired t-test with 95% CI on speedup percentage.
# Uses nameref arrays. Returns space-separated:
#   t_stat p_value ci_low ci_high mean_speedup
# Usage: result=$(paired_ttest_full tmac_array stock_array)
paired_ttest_full() {
    local -n _pta=$1  # T-MAC values
    local -n _ptb=$2  # Stock values
    local n=${#_pta[@]}
    if (( n < 2 )); then printf "0.00 1.0000 0.0 0.0 0.0"; return; fi

    awk -v n="$n" 'BEGIN {
        for (i = 1; i <= n; i++) {
            stock = ARGV[n + i]
            tmac = ARGV[i]
            speedup[i] = (tmac - stock) / stock * 100
        }

        sum = 0
        for (i = 1; i <= n; i++) sum += speedup[i]
        mean_sp = sum / n

        ss = 0
        for (i = 1; i <= n; i++) ss += (speedup[i] - mean_sp)^2
        sd_sp = sqrt(ss / (n - 1))
        se = sd_sp / sqrt(n)

        for (i = 1; i <= n; i++) d[i] = ARGV[i] - ARGV[n + i]
        sum_d = 0
        for (i = 1; i <= n; i++) sum_d += d[i]
        mean_d = sum_d / n
        ss_d = 0
        for (i = 1; i <= n; i++) ss_d += (d[i] - mean_d)^2
        sd_d = sqrt(ss_d / (n - 1))
        if (sd_d == 0 && mean_d == 0) { printf "0.00 1.0000 %.1f %.1f %.1f", mean_sp, mean_sp, mean_sp; exit }
        if (sd_d == 0) { printf "inf 0.0000 %.1f %.1f %.1f", mean_sp, mean_sp, mean_sp; exit }
        t = mean_d / (sd_d / sqrt(n))
        if (t < 0) t_abs = -t; else t_abs = t

        # Cornish-Fisher expansion: map t-statistic to z-score for small df
        # (from bench_db.py t_to_pvalue). For df>=30, normal approx is fine.
        df = n - 1
        if (df >= 30) z = t_abs
        else z = t_abs * (1 - 1/(4*df)) / sqrt(1 + t_abs*t_abs/(2*df))
        b1 = 0.319381530; b2 = -0.356563782; b3 = 1.781477937
        b4 = -1.821255978; b5 = 1.330274429; p_coeff = 0.2316419
        tt = 1.0 / (1.0 + p_coeff * z)
        phi = (1.0 / sqrt(2 * 3.14159265358979)) * exp(-z * z / 2.0)
        cdf = 1.0 - phi * (b1*tt + b2*tt^2 + b3*tt^3 + b4*tt^4 + b5*tt^5)
        p = 2.0 * (1.0 - cdf)

        if (df == 4) t_crit = 2.776
        else if (df == 9) t_crit = 2.262
        else if (df == 11) t_crit = 2.201
        else t_crit = 2.0 + 1.0 / df

        ci_lo = mean_sp - t_crit * se
        ci_hi = mean_sp + t_crit * se

        printf "%.2f %.4f %.1f %.1f %.1f", t_abs, p, ci_lo, ci_hi, mean_sp
    }' "${_pta[@]}" "${_ptb[@]}"
}

# Simplified paired t-test returning only p-value.
# Usage: p=$(paired_ttest tmac_array stock_array)
paired_ttest() {
    local result
    result=$(paired_ttest_full "$1" "$2")
    echo "$result" | awk '{print $2}'
}
