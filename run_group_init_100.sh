#!/usr/bin/env bash
set -euo pipefail

CODE=/workspace/variants/group_init
DATA=/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts
RESULTS=/auto/datasets/kondyurin/FinTexTS/runs/group_init_100_v1

cd "$CODE"
export PYTHONPATH="$CODE:$CODE/forecasting_task"

mkdir -p "$RESULTS"

# Prevent duplicate launches of this queue.
LOCK="$RESULTS/.queue.lock"

if ! mkdir "$LOCK" 2>/dev/null; then
    echo "Queue lock exists: $LOCK"
    echo "Check whether this queue is already running."
    exit 1
fi

trap 'rmdir "$LOCK"' EXIT

# Wait for the earlier queue and its current training process.
python -u - <<'PY'
from pathlib import Path
import time

old_script = "run_group_init.sh"
old_results = (
    "/auto/datasets/kondyurin/FinTexTS/runs/group_init_v1/"
)

def earlier_processes():
    found = []

    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue

        try:
            raw = (entry / "cmdline").read_bytes()
            args = raw.decode(errors="replace").split("\0")
            args = [arg for arg in args if arg]
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
        ):
            continue

        if not args:
            continue

        program = Path(args[0]).name

        old_driver = program in ("bash", "sh") and (
            any(
                Path(arg).name == old_script
                for arg in args[1:]
            )
            or (
                "-c" in args
                and any(
                    "/" + old_script in arg
                    for arg in args[1:]
                )
            )
        )

        old_worker = (
            "forecasting_task.run" in args
            and any(
                arg.startswith(old_results)
                for arg in args
            )
        )

        if old_driver or old_worker:
            found.append(entry.name)

    return found

while True:
    pids = earlier_processes()

    if not pids:
        break

    print(
        "WAIT: earlier group_init_v1 queue is running; PIDs="
        + ",".join(pids),
        flush=True,
    )

    time.sleep(15)

print(
    "Earlier queue finished. Starting the 100-stock comparison.",
    flush=True,
)
PY

# Check GPU availability.
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
print("GPU:", torch.cuda.get_device_name(0), flush=True)
PY

# Check that run.py supports the initialization argument.
python -m forecasting_task.run --help > "$RESULTS/runner_help.txt"

if ! grep -q -- '--initial_group_weights' "$RESULTS/runner_help.txt"; then
    echo "Missing --initial_group_weights in run.py"
    exit 1
fi

# Find all ticker files.
shopt -s nullglob
files=("$DATA"/*.parquet)

if [ "${#files[@]}" -ne 100 ]; then
    echo "Expected 100 ticker files, found ${#files[@]}"
    exit 1
fi

# Save the exact code used for this experiment.
cp "$CODE/forecasting_task/run.py" "$RESULTS/run.py.snapshot"
cp "$CODE/run_group_init_100.sh" "$RESULTS/run_group_init_100.sh.snapshot"

python - <<'PY' > "$RESULTS/runner_sha256.txt"
from pathlib import Path
import hashlib

path = Path(
    "/workspace/variants/group_init/forecasting_task/run.py"
)

print(hashlib.sha256(path.read_bytes()).hexdigest())
PY

seeds=(7 17 27)
initializations=(macro targetCompany)

mkdir -p "$RESULTS/full"

completed=0

for seed in "${seeds[@]}"; do
    for file in "${files[@]}"; do
        ticker=$(basename "$file" .parquet)

        for initialization in "${initializations[@]}"; do

            # Order: macro, sector, targetCompany, relatedCompany, filing.
            case "$initialization" in
                macro)
                    weights=(0.6 0.1 0.1 0.1 0.1)
                    ;;
                targetCompany)
                    weights=(0.1 0.1 0.6 0.1 0.1)
                    ;;
            esac

            destination="$RESULTS/full/$initialization/$ticker/seed$seed"

            if [ -e "$destination" ]; then
                echo "Directory already exists: $destination"
                echo "Refusing to overwrite existing results."
                exit 1
            fi

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
                # Verify the saved results and weight trajectories.
                if ! python - "$destination/output" "${weights[@]}" <<'PY'
from pathlib import Path
import json
import sys

import numpy as np

output = Path(sys.argv[1])
expected = np.asarray(sys.argv[2:], dtype=float)

summary = json.loads(
    (output / "summary.json").read_text()
)

assert (output / "best_checkpoint.pt").stat().st_size > 0

assert np.allclose(
    summary["initial_group_weights"],
    expected,
    atol=1e-6,
    rtol=0,
)

for name in ("best_group_weights", "final_group_weights"):
    values = np.asarray(summary[name], dtype=float)

    assert values.shape == (5,)
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert np.isclose(values.sum(), 1)

for name in (
    "best_val_mse",
    "best_test_mse",
    "best_test_mae",
):
    assert np.isfinite(summary[name])

records = [
    json.loads(line)
    for line in (output / "history.jsonl").read_text().splitlines()
    if line
]

assert len(records) == summary["epochs_completed"]

assert all(
    len(record["group_weights"]) == 5
    for record in records
)
PY
                then
                    echo "FAILED output verification: $destination"
                    exit 1
                fi

                completed=$((completed + 1))

                echo "OK init=$initialization ticker=$ticker seed=$seed completed=$completed/600"
            else
                echo "FAILED init=$initialization ticker=$ticker seed=$seed"
                tail -n 40 "$destination/train.log"
                exit 1
            fi
        done
    done
done

test "$completed" -eq 600

echo "ALL 600 RUNS COMPLETED"