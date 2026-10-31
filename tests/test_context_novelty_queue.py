"""Exercise the two checks and all 60 runs using fake CUDA and training."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_context_novelty_gate_60.sh"
TICKERS = "AAPL ABBV ABT ACN ADBE ADI AMAT AMD AMGN AMZN".split()


class ContextNoveltyQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = [os.environ.get("BASH_EXECUTABLE")]
        if os.name == "nt":
            candidates.append(r"C:\Program Files\Git\bin\bash.exe")
        candidates.append(shutil.which("bash"))
        for candidate in filter(None, candidates):
            try:
                if subprocess.run([candidate, "-c", "exit 0"], capture_output=True, timeout=10).returncode == 0:
                    cls.bash = candidate
                    return
            except (OSError, subprocess.TimeoutExpired):
                pass
        raise unittest.SkipTest("Bash unavailable; run bash -n run_context_novelty_gate_60.sh on server")

    def test_bash_syntax(self):
        result = subprocess.run([self.bash, "-n", str(SCRIPT)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def dispatch(self, root, mode="success", commit="a" * 40):
        data, results = root / "data", root / "results"
        data.mkdir()
        for ticker in [*TICKERS, "NVDA"]:
            if mode != "missing_data" or ticker != "NVDA":
                (data / f"{ticker}.parquet").touch()
        text = SCRIPT.read_text().replace(
            "DATA=/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts",
            "DATA=" + shlex.quote(data.as_posix()),
        ).replace(
            "RESULTS=/auto/datasets/kondyurin/FinTexTS/runs/context_novelty_gate_v1",
            "RESULTS=" + shlex.quote(results.as_posix()),
        )
        script = root / SCRIPT.name
        script.write_text(text, newline="\n")
        calls = root / "calls.bin"
        if mode == "existing":
            destination = results / "full" / "context_novelty" / "AMZN" / "seed27"
            destination.mkdir(parents=True)
            (destination / "train.log").write_text("preserve user log")
        harness = r'''
python() {
    if [[ "$OMP_NUM_THREADS" != 2 || "$MKL_NUM_THREADS" != 2 || "$PYTHONPATH" != "$PWD:$PWD/forecasting_task"* ]]; then
        return 1
    fi
    if [[ "$1" == - ]]; then
        if [[ "$MOCK_MODE" == no_cuda ]]; then echo 'mock CUDA unavailable' >&2; return 1; fi
        return 0
    fi
    printf '%s\0' "$@" >> "$MOCK_CALLS"
    printf '\0' >> "$MOCK_CALLS"
    if [[ "$MOCK_MODE" == failure ]]; then echo 'mock training failure'; return 9; fi
    local output=''
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == --logdir ]]; then output=$2; break; fi
        shift
    done
    if [[ "$MOCK_MODE" == success ]]; then
        mkdir -p -- "$output"
        printf '{}\n' > "$output/summary.json"
        printf 'mock checkpoint\n' > "$output/best_checkpoint.pt"
    fi
    echo 'mock training only'
}
git() { return 1; }
export -f python git
"$BASH" "$MOCK_SCRIPT"
'''
        env = os.environ.copy()
        env.update(MOCK_SCRIPT=script.as_posix(), MOCK_MODE=mode, MOCK_CALLS=calls.as_posix())
        if commit is None:
            env.pop("CODE_COMMIT", None)
        else:
            env["CODE_COMMIT"] = commit
        result = subprocess.run([self.bash, "-c", harness], env=env, capture_output=True, text=True, timeout=60)
        records = []
        if calls.exists():
            records = [[token.decode() for token in record.split(b"\0")]
                       for record in calls.read_bytes().split(b"\0\0") if record]
        return result, records, results, script

    def test_exact_60_plus_two_one_epoch_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            result, records, results, script = self.dispatch(Path(directory))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(records), 62)
            self.assertEqual(result.stdout.count("START "), 62)
            self.assertEqual(result.stdout.count("OK "), 62)
            self.assertIn("completed=60/60", result.stdout)
            self.assertTrue(result.stdout.rstrip().endswith("ALL 60 RUNS COMPLETED"))
            self.assertEqual((results / SCRIPT.name).read_bytes(), script.read_bytes())
            self.assertEqual((results / "code_commit.txt").read_text().strip(), "a" * 40)
            self.assertEqual((results / "selected_tickers.txt").read_text().splitlines(), TICKERS)
            observed = set()
            for index, record in enumerate(records):
                self.assertEqual(record[:3], ["-u", "-m", "forecasting_task.run"])
                flags = dict(zip(record[3::2], record[4::2]))
                stage, condition, ticker, seed, output = flags["--logdir"].replace("\\", "/").split("/")[-5:]
                self.assertEqual(output, "output")
                self.assertEqual(flags["--group_gate"], condition)
                self.assertEqual(flags["--data_path"], ticker + ".parquet")
                self.assertEqual(flags["--seed"], seed.removeprefix("seed"))
                expected = {"model_type": "FiLM", "seq_len": "64", "label_len": "16", "pred_len": "3",
                            "batch_size": "32", "text_num_layers": "1", "text_hidden_size": "64",
                            "text_weight": "0.1", "group_weight_mode": "fixed_uniform", "fusion_mode": "blend",
                            "lr": "0.001", "patience": "10", "code_commit": "a" * 40,
                            "used_col_prefixes": "macro,sector,targetCompany,relatedCompany,filing"}
                for name, value in expected.items():
                    self.assertEqual(flags["--" + name], value)
                if index < 2:
                    self.assertEqual((stage, ticker, seed), ("checks", "NVDA", "seed7"))
                    self.assertEqual(condition, ("static", "context_novelty")[index])
                    self.assertEqual(flags["--num_epoch"], "1")
                else:
                    self.assertEqual(stage, "full")
                    self.assertEqual(flags["--num_epoch"], "50")
                    observed.add((condition, ticker, seed))
            self.assertEqual(observed, {(condition, ticker, f"seed{seed}")
                                        for condition in ("static", "context_novelty")
                                        for ticker in TICKERS for seed in (7, 17, 27)})
            self.assertEqual(len(observed), 60)

    def test_preflight_and_check_failures_stop_before_full_queue(self):
        for mode, commit, status, message in (
            ("missing_data", "a" * 40, 1, "Missing data"),
            ("no_cuda", "a" * 40, 1, "mock CUDA unavailable"),
            ("missing", None, 1, "Set CODE_COMMIT"),
            ("missing", "bad", 1, "full commit SHA"),
            ("existing", "a" * 40, 1, "Run directory already exists"),
            ("failure", "a" * 40, 9, "mock training failure"),
            ("missing", "a" * 40, 1, "FAILED missing/empty"),
        ):
            with self.subTest(mode=mode, commit=commit), tempfile.TemporaryDirectory() as directory:
                result, records, results, _ = self.dispatch(Path(directory), mode, commit)
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertIn(message, result.stderr)
                self.assertNotIn("OK ", result.stdout)
                self.assertNotIn("ALL 60 RUNS COMPLETED", result.stdout)
                self.assertEqual(len(records), int(mode == "failure" or message == "FAILED missing/empty"))
                if mode == "existing":
                    self.assertEqual((results / "full/context_novelty/AMZN/seed27/train.log").read_text(), "preserve user log")
