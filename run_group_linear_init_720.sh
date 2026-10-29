#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_PATH="$CODE/$(basename -- "${BASH_SOURCE[0]}")"
DATA=/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts
RESULTS=/auto/datasets/kondyurin/FinTexTS/runs/group_linear_init_720_v1
export PYTHONPATH="$CODE:$CODE/forecasting_task${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$CODE"

shopt -s nullglob
files=()
for file in "$DATA"/*.parquet; do
    if [[ -f "$file" ]]; then
        files+=("$file")
    fi
done
if [[ ${#files[@]} -ne 100 ]]; then
    printf 'Expected exactly 100 parquet files in %s; found %d.\n' "$DATA" "${#files[@]}" >&2
    exit 1
fi
selected_files=("${files[@]:0:10}")
selected_tickers=()
for file in "${selected_files[@]}"; do
    ticker=${file##*/}
    selected_tickers+=("${ticker%.parquet}")
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
# Preserve metadata too: an existing file must describe this same experiment.
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
record_text "$RESULTS/selected_tickers.txt" "$(printf '%s\n' "${selected_tickers[@]}")"
script_copy="$RESULTS/$(basename -- "$SCRIPT_PATH")"
if [[ -e "$script_copy" || -L "$script_copy" ]]; then
    if [[ ! -f "$script_copy" ]] || ! cmp -s -- "$SCRIPT_PATH" "$script_copy"; then
        echo 'Existing script copy differs. Use a new RESULTS directory.' >&2
        exit 1
    fi
else
    (set -o noclobber; cat -- "$SCRIPT_PATH" > "$script_copy")
fi

completed=0
for initialization in uniform macro targetCompany; do
    case "$initialization" in
        uniform)
            weights=(0.2 0.2 0.2 0.2 0.2)
            queue_files=("${files[@]}")
            ;;
        macro)
            weights=(0.6 0.1 0.1 0.1 0.1)
            queue_files=("${selected_files[@]}")
            ;;
        targetCompany)
            weights=(0.1 0.1 0.6 0.1 0.1)
            queue_files=("${selected_files[@]}")
            ;;
    esac
    for fusion_mode in blend residual; do
        for seed in 7 17 27; do
            for file in "${queue_files[@]}"; do
                ticker=${file##*/}
                ticker=${ticker%.parquet}
                destination="$RESULTS/full/$initialization/$fusion_mode/$ticker/seed$seed"
                # mkdir without -p for the run directory also prevents concurrent overwrites.
                mkdir -p -- "${destination%/*}"
                if ! mkdir -- "$destination"; then
                    printf 'Run directory already exists or cannot be created: %s\n' "$destination" >&2
                    exit 1
                fi
                printf 'START initialization=%s fusion_mode=%s ticker=%s seed=%s completed=%d/720\n' \
                    "$initialization" "$fusion_mode" "$ticker" "$seed" "$completed"
                if python -u -m forecasting_task.run \
                    --root_path "$DATA" \
                    --data_path "$ticker.parquet" \
                    --model_type FiLM \
                    --used_col_prefixes macro,sector,targetCompany,relatedCompany,filing \
                    --seq_len 64 \
                    --label_len 16 \
                    --pred_len 3 \
                    --batch_size 32 \
                    --text_num_layers 1 \
                    --text_hidden_size 64 \
                    --text_weight 0.1 \
                    --initial_group_weights "${weights[@]}" \
                    --lr 0.001 \
                    --num_epoch 50 \
                    --patience 10 \
                    --seed "$seed" \
                    --fusion_mode "$fusion_mode" \
                    --logdir "$destination/output" \
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
                completed=$((completed + 1))
                printf 'OK initialization=%s fusion_mode=%s ticker=%s seed=%s completed=%d/720\n' \
                    "$initialization" "$fusion_mode" "$ticker" "$seed" "$completed"
            done
        done
    done
done

if [[ "$completed" -ne 720 ]]; then
    printf 'Unexpected run count: %d/720\n' "$completed" >&2
    exit 1
fi
echo 'ALL 720 RUNS COMPLETED'
