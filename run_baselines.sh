#!/usr/bin/env bash
set -eu

cd /workspace/FinTexTS

DATA=/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts
RESULTS=/auto/datasets/kondyurin/FinTexTS/runs/baselines_full_v1

export PYTHONPATH=/workspace/FinTexTS:/workspace/FinTexTS/forecasting_task

mkdir -p "$RESULTS"

# Record the code version and the actual container environment.
git rev-parse HEAD > "$RESULTS/code_commit.txt"
python -m pip freeze > "$RESULTS/environment.txt"

if [ "$(find "$DATA" -maxdepth 1 -name '*.parquet' -type f | wc -l)" -ne 100 ]; then
    echo "Expected exactly 100 ticker files."
    exit 1
fi

for model in \
    Autoformer Crossformer DLinear FiLM Informer iTransformer \
    Nonstationary_Transformer PatchTST Reformer TiDE Transformer TSMixer
do
    for variant in price with_text
    do
        if [ "$variant" = price ]; then
            weight=0
        else
            weight=0.1
        fi

        for seed in 7 17 27
        do
            for file in "$DATA"/*.parquet
            do
                ticker=$(basename "$file" .parquet)
                destination="$RESULTS/$model/$variant/$ticker/seed$seed"

                # Preserve previous attempts and their logs.
                if [ -e "$destination" ]; then
                    echo "Existing directory: $destination"
                    echo "Use a new RESULTS directory for a fresh queue."
                    exit 1
                fi

                mkdir -p "$destination"

                echo "START model=$model variant=$variant ticker=$ticker seed=$seed"

                if python -u -m forecasting_task.run \
                    --root_path "$DATA" \
                    --data_path "$ticker.parquet" \
                    --model_type "$model" \
                    --used_col_prefixes macro,sector,targetCompany,relatedCompany,filing \
                    --seq_len 64 \
                    --label_len 16 \
                    --pred_len 3 \
                    --batch_size 32 \
                    --num_epoch 10 \
                    --patience 5 \
                    --lr 0.001 \
                    --text_weight "$weight" \
                    --seed "$seed" \
                    --logdir "$destination/output" \
                    > "$destination/train.log" 2>&1
                then
                    echo "OK model=$model variant=$variant ticker=$ticker seed=$seed"
                else
                    status=$?
                    echo "FAILED exit=$status model=$model variant=$variant ticker=$ticker seed=$seed"
                    tail -n 30 "$destination/train.log"
                    exit "$status"
                fi
            done
        done
    done
done

echo "ALL RUNS COMPLETED"