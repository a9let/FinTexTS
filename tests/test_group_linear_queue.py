"""Exercise the Bash dispatcher with fake CUDA/training, never real models."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_group_linear_init_720.sh"
WEIGHTS = {
    "uniform": ["0.2"] * 5,
    "macro": ["0.6", "0.1", "0.1", "0.1", "0.1"],
    "targetCompany": ["0.1", "0.1", "0.6", "0.1", "0.1"],
}


class LinearQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = [os.environ.get("BASH_EXECUTABLE")]
        if os.name == "nt":
            candidates.append(r"C:\Program Files\Git\bin\bash.exe")
        candidates.append(shutil.which("bash"))
        for candidate in filter(None, candidates):
            try:
                result = subprocess.run([candidate, "-c", "exit 0"], capture_output=True, timeout=10)
                if result.returncode == 0:
                    cls.bash = candidate
                    return
            except (OSError, subprocess.TimeoutExpired):
                pass
        raise unittest.SkipTest("Bash unavailable; run bash -n run_group_linear_init_720.sh on the server")

    def test_bash_syntax(self):
        result = subprocess.run([self.bash, "-n", str(SCRIPT)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def dispatch(self, root, mode="success", count=100, code_commit="a" * 40):
        data = root / "data"
        results = root / "results"
        data.mkdir()
        # Deliberately create in reverse order; the dispatcher must select alphabetically.
        for index in reversed(range(count)):
            (data / f"T{index:03}.parquet").touch()
        text = SCRIPT.read_text().replace(
            "DATA=/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts",
            "DATA=" + shlex.quote(data.as_posix()),
        ).replace(
            "RESULTS=/auto/datasets/kondyurin/FinTexTS/runs/group_linear_init_720_v1",
            "RESULTS=" + shlex.quote(results.as_posix()),
        )
        script = root / SCRIPT.name
        script.write_text(text, newline="\n")
        calls = root / "calls.bin"
        if mode == "success":
            # Fake completed artifacts; shell functions below mock training and mkdir.
            for initialization in WEIGHTS:
                for fusion in ("blend", "residual"):
                    for seed in (7, 17, 27):
                        for index in range(100 if initialization == "uniform" else 10):
                            output = results / "full" / initialization / fusion / f"T{index:03}" / f"seed{seed}" / "output"
                            output.mkdir(parents=True)
                            (output / "summary.json").write_text("{}")
                            (output / "best_checkpoint.pt").write_text("mock checkpoint")
        elif mode == "existing":
            destination = results / "full" / "uniform" / "blend" / "T000" / "seed7"
            destination.mkdir(parents=True)
            (destination / "train.log").write_text("preserve this log")
        harness = r'''
python() {
    if [[ "$OMP_NUM_THREADS" != 2 || "$MKL_NUM_THREADS" != 2 || "$PYTHONPATH" != "$PWD:$PWD/forecasting_task"* ]]; then
        echo 'incorrect thread/import configuration' >&2
        return 1
    fi
    if [[ "$1" == - ]]; then
        if [[ "$MOCK_MODE" == no_cuda ]]; then
            echo 'mock CUDA unavailable' >&2
            return 1
        fi
        return 0
    fi
    printf '%s\0' "$@" >> "$MOCK_CALLS"
    printf '\0' >> "$MOCK_CALLS"
    if [[ "$MOCK_MODE" == failure ]]; then
        echo 'mock training failure'
        return 9
    fi
    echo 'mock training only'
}
git() { return 1; }
export -f python git
if [[ "$MOCK_MODE" == success ]]; then
    # All output directories above are fixtures, not prior real runs.
    mkdir() { return 0; }
    export -f mkdir
fi
"$BASH" "$MOCK_SCRIPT"
'''
        env = os.environ.copy()
        env.update(MOCK_SCRIPT=script.as_posix(), MOCK_MODE=mode, MOCK_CALLS=calls.as_posix())
        if code_commit is None:
            env.pop("CODE_COMMIT", None)
        else:
            env["CODE_COMMIT"] = code_commit
        result = subprocess.run([self.bash, "-c", harness], env=env, capture_output=True, text=True, timeout=60)
        records = []
        if calls.exists():
            records = [[item.decode() for item in record.split(b"\0")]
                       for record in calls.read_bytes().split(b"\0\0") if record]
        return result, records, results, script

    def test_dispatches_exact_720_configurations_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            result, records, results, script = self.dispatch(Path(directory))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(records), 720)
            self.assertEqual(result.stdout.count("START "), 720)
            self.assertEqual(result.stdout.count("OK "), 720)
            self.assertIn("completed=720/720", result.stdout)
            self.assertTrue(result.stdout.rstrip().endswith("ALL 720 RUNS COMPLETED"))
            self.assertEqual((results / "selected_tickers.txt").read_text().splitlines(),
                             [f"T{index:03}" for index in range(10)])
            self.assertEqual((results / "code_commit.txt").read_text().strip(), "a" * 40)
            self.assertEqual((results / SCRIPT.name).read_bytes(), script.read_bytes())
            observed = set()
            counts = dict.fromkeys(WEIGHTS, 0)
            for record in records:
                self.assertEqual(record[:3], ["-u", "-m", "forecasting_task.run"])
                flags = {}
                for token in record[3:]:
                    if token.startswith("--"):
                        key = token
                        flags[key] = []
                    else:
                        flags[key].append(token)
                parts = flags["--logdir"][0].replace("\\", "/").split("/")
                initialization, fusion, ticker, seed, output = parts[-5:]
                self.assertEqual(output, "output")
                self.assertEqual(flags["--fusion_mode"], [fusion])
                self.assertEqual(flags["--seed"], [seed.removeprefix("seed")])
                self.assertEqual(flags["--data_path"], [ticker + ".parquet"])
                self.assertEqual(flags["--initial_group_weights"], WEIGHTS[initialization])
                expected_flags = {"model_type": "FiLM", "seq_len": "64", "label_len": "16", "pred_len": "3",
                                  "batch_size": "32", "text_num_layers": "1", "text_hidden_size": "64",
                                  "text_weight": "0.1", "lr": "0.001", "num_epoch": "50", "patience": "10",
                                  "used_col_prefixes": "macro,sector,targetCompany,relatedCompany,filing"}
                for key, value in expected_flags.items():
                    self.assertEqual(flags["--" + key], [value])
                self.assertIn(fusion, ("blend", "residual"))
                self.assertIn(seed, ("seed7", "seed17", "seed27"))
                if initialization != "uniform":
                    self.assertIn(ticker, [f"T{index:03}" for index in range(10)])
                counts[initialization] += 1
                observed.add((initialization, fusion, ticker, seed))
            self.assertEqual(len(observed), 720)
            self.assertEqual(counts, {"uniform": 600, "macro": 60, "targetCompany": 60})

    def test_stops_on_preflight_existing_run_failure_and_missing_artifacts(self):
        for mode, count, commit, expected_status, message in (
            ("missing", 99, "a" * 40, 1, "Expected exactly 100"),
            ("no_cuda", 100, "a" * 40, 1, "mock CUDA unavailable"),
            ("missing", 100, None, 1, "Set CODE_COMMIT"),
            ("existing", 100, "a" * 40, 1, "Run directory already exists"),
            ("failure", 100, "a" * 40, 9, "mock training failure"),
            ("missing", 100, "a" * 40, 1, "FAILED missing/empty"),
        ):
            with self.subTest(mode=mode, count=count, commit=commit), tempfile.TemporaryDirectory() as directory:
                result, records, results, _ = self.dispatch(Path(directory), mode, count, commit)
                self.assertEqual(result.returncode, expected_status, result.stderr)
                self.assertIn(message, result.stderr)
                self.assertNotIn("OK ", result.stdout)
                self.assertNotIn("ALL 720 RUNS COMPLETED", result.stdout)
                self.assertEqual(len(records), 1 if mode == "failure" or "FAILED missing/empty" in message else 0)
                if mode == "existing":
                    log = results / "full" / "uniform" / "blend" / "T000" / "seed7" / "train.log"
                    self.assertEqual(log.read_text(), "preserve this log")
