#!/usr/bin/env bash
set -euo pipefail

CODE=/workspace/variants/group_init
DATA=/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts
RESULTS=/auto/datasets/kondyurin/FinTexTS/runs/group_init_v1

cd "$CODE"
export PYTHONPATH="$CODE:$CODE/forecasting_task"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'

tickers=(AAPL AMZN NVDA JPM XOM)
seeds=(7 17 27)
initializations=(uniform macro sector targetCompany relatedCompany filing)

for ticker in "${tickers[@]}"; do
    test -f "$DATA/$ticker.parquet"
done

if [ -e "$RESULTS/full" ]; then
    echo "Results already exist: $RESULTS/full"
    exit 1
fi

mkdir -p "$RESULTS/full"

for seed in "${seeds[@]}"; do
    for ticker in "${tickers[@]}"; do
        for initialization in "${initializations[@]}"; do

            case "$initialization" in
                uniform)        weights=(0.2 0.2 0.2 0.2 0.2) ;;
                macro)          weights=(0.6 0.1 0.1 0.1 0.1) ;;
                sector)         weights=(0.1 0.6 0.1 0.1 0.1) ;;
                targetCompany)  weights=(0.1 0.1 0.6 0.1 0.1) ;;
                relatedCompany) weights=(0.1 0.1 0.1 0.6 0.1) ;;
                filing)         weights=(0.1 0.1 0.1 0.1 0.6) ;;
            esac

            destination="$RESULTS/full/$initialization/$ticker/seed$seed"
            mkdir -p "$destination"

            echo "START init=$initialization ticker=$ticker seed=$seed"

            if python -u -m forecasting_task.run \
                --root_path "$DATA" \
                --data_path "$ticker.parquet" \
                --model_type FiLM \
                --used_col_prefixes macro,sector,targetCompany,relatedCompany,filing \
                --initial_group_weights "${weights[@]}" \
                --seq_len 64 \
                --label_len 16 \
                --pred_len 3 \
                --batch_size 32 \
                --num_epoch 50 \
                --patience 10 \
                --lr 0.001 \
                --text_weight 0.1 \
                --seed "$seed" \
                --logdir "$destination/output" \
                > "$destination/train.log" 2>&1
            then
                test -s "$destination/output/summary.json"
                test -s "$destination/output/best_checkpoint.pt"
                echo "OK init=$initialization ticker=$ticker seed=$seed"
            else
                echo "FAILED init=$initialization ticker=$ticker seed=$seed"
                tail -n 40 "$destination/train.log"
                exit 1
            fi

        done
    done
done

echo "ALL 90 RUNS COMPLETED"