#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_PATH="$CODE/$(basename -- "${BASH_SOURCE[0]}")"
DATA=/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts
RESULTS=/auto/datasets/kondyurin/FinTexTS/runs/context_novelty_gate_v1
export PYTHONPATH="$CODE:$CODE/forecasting_task${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$CODE"

tickers=(AAPL ABBV ABT ACN ADBE ADI AMAT AMD AMGN AMZN)
for ticker in "${tickers[@]}" NVDA; do
    if [[ ! -f "$DATA/$ticker.parquet" ]]; then
        printf 'Missing data: %s/%s.parquet\n' "$DATA" "$ticker" >&2
        exit 1
    fi
done
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; no training runs were started.")
torch.empty(1, device="cuda")
print(f"CUDA ready: {torch.cuda.get_device_name(0)}", flush=True)
PY

code_commit=${CODE_COMMIT:-}
if [[ -z "$code_commit" ]]; then
    if ! code_commit=$(git -C "$CODE" rev-parse --verify HEAD 2>/dev/null); then
        echo 'Git metadata is unavailable. Set CODE_COMMIT to the full code commit SHA.' >&2
        exit 1
    fi
fi
if [[ ! "$code_commit" =~ ^([[:xdigit:]]{40}|[[:xdigit:]]{64})$ ]]; then
    echo 'CODE_COMMIT must be a full commit SHA (40 or 64 hexadecimal characters).' >&2
    exit 1
fi

mkdir -p -- "$RESULTS"
record_text() {
    local path=$1 content=$2
    if [[ -e "$path" || -L "$path" ]]; then
        if [[ ! -f "$path" ]] || ! printf '%s\n' "$content" | cmp -s - "$path"; then
            printf 'Existing metadata differs: %s. Use a new RESULTS directory.\n' "$path" >&2
            exit 1
        fi
    else
        (set -o noclobber; printf '%s\n' "$content" > "$path")
    fi
}
record_text "$RESULTS/code_commit.txt" "$code_commit"
record_text "$RESULTS/selected_tickers.txt" "$(printf '%s\n' "${tickers[@]}")"
script_copy="$RESULTS/$(basename -- "$SCRIPT_PATH")"
if [[ -e "$script_copy" || -L "$script_copy" ]]; then
    if [[ ! -f "$script_copy" ]] || ! cmp -s -- "$SCRIPT_PATH" "$script_copy"; then
        echo 'Existing script copy differs. Use a new RESULTS directory.' >&2
        exit 1
    fi
else
    (set -o noclobber; cat -- "$SCRIPT_PATH" > "$script_copy")
fi

# Check every destination before starting any training. mkdir below also guards
# against concurrent launches after this preflight.
check_destination() {
    if [[ -e "$1" || -L "$1" ]]; then
        printf 'Run directory already exists: %s\n' "$1" >&2
        exit 1
    fi
}
for condition in static context_novelty; do
    check_destination "$RESULTS/checks/$condition/NVDA/seed7"
    for ticker in "${tickers[@]}"; do
        for seed in 7 17 27; do
            check_destination "$RESULTS/full/$condition/$ticker/seed$seed"
        done
    done
done

completed=0
run_one() {
    local stage=$1 condition=$2 ticker=$3 seed=$4 epochs=$5
    local destination="$RESULTS/$stage/$condition/$ticker/seed$seed"
    mkdir -p -- "${destination%/*}"
    if ! mkdir -- "$destination"; then
        printf 'Run directory already exists or cannot be created: %s\n' "$destination" >&2
        exit 1
    fi
    printf 'START stage=%s condition=%s ticker=%s seed=%s completed=%d/60\n' \
        "$stage" "$condition" "$ticker" "$seed" "$completed"
    if python -u -m forecasting_task.run \
        --root_path "$DATA" --data_path "$ticker.parquet" \
        --model_type FiLM \
        --used_col_prefixes macro,sector,targetCompany,relatedCompany,filing \
        --seq_len 64 --label_len 16 --pred_len 3 --batch_size 32 \
        --text_num_layers 1 --text_hidden_size 64 \
        --group_gate "$condition" --text_weight 0.1 \
        --group_weight_mode fixed_uniform --fusion_mode blend \
        --lr 0.001 --num_epoch "$epochs" --patience 10 --seed "$seed" \
        --code_commit "$code_commit" --logdir "$destination/output" \
        > "$destination/train.log" 2>&1
    then
        if [[ ! -s "$destination/output/summary.json" || ! -s "$destination/output/best_checkpoint.pt" ]]; then
            printf 'FAILED missing/empty summary.json or best_checkpoint.pt: %s\n' "$destination" >&2
            tail -n 30 -- "$destination/train.log" >&2
            exit 1
        fi
    else
        status=$?
        printf 'FAILED exit=%d: %s\n' "$status" "$destination" >&2
        tail -n 30 -- "$destination/train.log" >&2
        exit "$status"
    fi
    if [[ "$stage" == full ]]; then
        completed=$((completed + 1))
    fi
    printf 'OK stage=%s condition=%s ticker=%s seed=%s completed=%d/60\n' \
        "$stage" "$condition" "$ticker" "$seed" "$completed"
}

for condition in static context_novelty; do
    run_one checks "$condition" NVDA 7 1
done
for condition in static context_novelty; do
    for ticker in "${tickers[@]}"; do
        for seed in 7 17 27; do
            run_one full "$condition" "$ticker" "$seed" 50
        done
    done
done
if [[ "$completed" -ne 60 ]]; then
    printf 'Unexpected run count: %d/60\n' "$completed" >&2
    exit 1
fi
echo 'ALL 60 RUNS COMPLETED'
