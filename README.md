# FinTexTS

This repository contains code for forecasting with financial text and time-series data. It is organized into two parts: **data generation** and **forecasting experiments**.

## Structure

### 1) Data generation
Only a lightweight version of the data generation pipeline is included. Sample data can be created using the scripts in `make_dataset/`, for example:

- `make_dataset/company_summary.py`
- `make_dataset/filing_parser.py`
- `make_dataset/macro-sector_summary.py`
- `make_dataset/tag_company_news.py`
- `make_dataset/tag_news.py`

For access to the full dataset, refer to the following Hugging Face page:  
<https://huggingface.co/EXAONE-BI>  
[Reference link](https://huggingface.co/EXAONE-BI)

Raw news data was sourced from the FNSPID news dataset:
<https://github.com/Zdong104/FNSPID_Financial_News_Dataset>

### 2) Forecasting experiments
The entry point for forecasting experiments is `forecasting_task/run.py`. Before running it, you must generate text embeddings and create a per‑ticker parquet file.

#### (1) Text embedding generation
BERT embeddings are the default. Depending on the embedding model, columns are created as `*_emb1 ... *_embN`.

#### (2) Per‑ticker parquet creation
Create a parquet file **per ticker** with the following columns and save it to `data/fintexts/{TICKER}.parquet`.

- `date`, `ticker`, `open`, `high`, `low`, `close`
- `macro_emb1`, `macro_emb2`, ...
- `sector_emb1`, `sector_emb2`, ...
- `targetCompany_emb1`, `targetCompany_emb2`, ...
- `relatedCompany_emb1`, `relatedCompany_emb2`, ...
- `filing_emb1`, `filing_emb2`, ...

#### (3) Run example
```bash
python forecasting_task/run.py \
  --root_path data/ \
  --data_path fintexts/NVDA.parquet
```

### Independent group MLP fusion

The text branch consumes the last day's separate embeddings in this order:
`macro, sector, targetCompany, relatedCompany, filing` (`[batch, 5, 384]`).
Each group has its own MLP. `--text_num_layers 1` uses `Linear(384, pred_len)`;
larger values use `--text_hidden_size` hidden units and ReLU between linear layers.
The dataset preserves group embeddings and fits the price scaler on training dates only.

Group logits `G` start at `log(initial_weights)`, so `softmax(G)` initially assigns
the chosen group weights (uniform 0.2 per group by default). Set them with
`--initial_group_weights 0.6 0.1 0.1 0.1 0.1`, in `--used_col_prefixes` order.
The count must match the selected groups; values must be finite and strictly
positive, and their sum must equal 1 within absolute tolerance `1e-6`.
Initializing `G` consumes no random numbers and happens after head construction;
the same seed therefore gives identical initial FiLM and head parameters across
group initializations. A group is present when its original embedding has absolute sum
at least `1e-6`. Its output is masked **after** the MLP; missing groups contribute
zero without redistributing their weights. The weighted sum `delta` has one
correction per forecast horizon, shared across all four OHLC channels.

`alpha = sigmoid(alpha_logit)` is trained jointly with FiLM, the MLPs and `G`.
`--text_weight` sets its initial value (default 0.1, strictly between 0 and 1).
Choose `--fusion_mode`:

- `blend` (default): `(1-alpha) * price_output + alpha * (last_price + delta)`.
  If all groups are absent, the entire text prediction is zero, including last price.
- `residual`: `price_output + alpha * delta`. If all groups are absent, this equals
  the price forecast exactly; last price is never added in this mode.

From the repository root in PowerShell, set the existing TS modules' import path:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\forecasting_task"
```

Two otherwise identical one-epoch trial commands (require the prepared parquet data):

```powershell
python -m forecasting_task.run --model_type FiLM --root_path data/ --data_path fintexts/NVDA.parquet --seq_len 64 --label_len 16 --pred_len 3 --batch_size 32 --text_num_layers 2 --text_hidden_size 64 --text_weight 0.1 --seed 7 --num_epoch 1 --patience 1 --fusion_mode blend --logdir logs/group_mlp_blend_smoke
python -m forecasting_task.run --model_type FiLM --root_path data/ --data_path fintexts/NVDA.parquet --seq_len 64 --label_len 16 --pred_len 3 --batch_size 32 --text_num_layers 2 --text_hidden_size 64 --text_weight 0.1 --seed 7 --num_epoch 1 --patience 1 --fusion_mode residual --logdir logs/group_mlp_residual_smoke
```

Use an empty/new log directory for each run. `args.yaml` and `best_checkpoint.pt`
record the fusion mode, group order, MLP structure and initial settings.
The checkpoint stores both model state dictionaries (including all group MLPs,
`G` and `alpha_logit`), optimizer state, and train scaler statistics. Reconstruct
the text branch with `TextModel(**checkpoint["args"]["text_model_config"])`, then
load `checkpoint["text_model_state_dict"]`. Reconstruct FiLM using checkpoint
arguments and load `checkpoint["model_state_dict"]`; call `eval()` on both models
for inference and use the saved scaler statistics to scale incoming prices.

Each `history.jsonl` epoch records `text_alpha` and named `group_weights`.
`summary.json` separates `best_text_alpha` / `best_group_weights` (the best
validation checkpoint) from `final_text_alpha` / `final_group_weights` (the last
completed epoch). Existing MSE/MAE, timing, chronological splits and early stopping
are preserved.

`initial_group_weights` in `args.yaml`, `summary.json` and checkpoint arguments
is an ordered list of the actual initial softmax values, captured before the
optimizer is created. `initial_group_logits` in the arguments records the actual
starting logits. `text_model_config.initial_group_weights` retains the requested
weights for reconstruction; tiny float32 rounding differences are expected.
Initial values are separate from the named best/final group weight dictionaries.

### Linear heads without learned group weights

Select `--group_weight_mode fixed_uniform --text_num_layers 1` for the new
variant. The existing `learnable` mode remains available for previous experiments.
The fixed variant requires all five groups in the order
`macro, sector, targetCompany, relatedCompany, filing`. Each head is independently
initialized as `Linear(384, pred_len)`, with no hidden layer or activation.
`text_hidden_size` does not affect these heads. The fixed model has no `G`
parameter or group softmax, including in its optimizer and checkpoint.

Using only the last day of the text window, it computes:

```text
mask[g] = abs(embedding[g]).sum(-1) >= 1e-6
delta = sum(mask[g] * head[g](embedding[g])) / 5
alpha = sigmoid(alpha_logit)
```

The mask applies after each head, including its bias. Missing groups contribute
zero; the denominator always remains five. Each horizon correction is shared
by all four OHLC channels. Fusion is unchanged:

- `blend`: `text_prediction = last_price + delta`, with the entire text prediction
  set to zero when all groups are absent;
  `output = (1-alpha)*price_output + alpha*text_prediction`.
- `residual`: `output = price_output + alpha*delta`, exactly equal to the price
  forecast when all groups are absent. Last price is not added.

`--text_weight` initializes alpha and must be strictly between zero and one.
Setting alpha and choosing the fusion mode consume no random numbers, so a given
seed produces identical initial FiLM and head parameters for alpha starts
0.01, 0.1 and 0.9 in both modes. FiLM, the heads and alpha train jointly with
the existing Adam, validation checkpoint selection and early stopping. Scaling,
chronological splits, metrics and timing remain unchanged.

`--initial_group_weights` may be omitted or set to five uniform values; nonuniform
values are explicitly rejected in this mode. Logs record
`group_weight_mode: fixed_uniform` and `fixed_group_coefficients` (0.2 per group),
without labeling these constants as learned best/final weights. Arguments,
summary and checkpoint arguments retain the actual `initial_text_alpha`;
`best_text_alpha` and `final_text_alpha` in the summary refer to the selected
validation checkpoint and last epoch respectively. Epoch history records alpha.

For restoration, call
`build_text_model(checkpoint["args"]["text_model_config"], checkpoint["args"]["group_weight_mode"])`
from `forecasting_task.models.GroupText`, then load `text_model_state_dict`.
The FiLM parameters, Adam state and train scaler statistics are also saved.
Older checkpoints without a mode use `learnable`.

A checkpoint from the old one-layer model can be migrated while preserving
predictions: for each group with old `w = softmax(G)[g]`, copy
`W_new[g] = 5*w*W_old[g]`, `b_new[g] = 5*w*b_old[g]` and the same alpha logit.
Drop `G` from the new state dictionary. This preserves the forward calculation
with masking; it does not assert identical training trajectories from scratch.

### Server experiment: 720 runs without learned group weights

Run `bash run_group_no_weights_720.sh` in the existing CUDA-enabled Docker.
The fixed-uniform server experiment uses
`/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts` for data and
`/auto/datasets/kondyurin/FinTexTS/runs/group_no_weights_alpha_v1` for results.
Its settings are FiLM, sequence length 64, label length 16, prediction length 3,
batch size 32, `text_num_layers=1`, learning rate 0.001, at most 50 epochs and
early stopping patience 10. Alpha labels and initial values are:

| Label | Initial alpha | Tickers | Seeds | Fusion modes | Runs |
| --- | --- | --- | --- | --- | --- |
| alpha010 | 0.1 | All 100 | 7, 17, 27 | blend, residual | 600 |
| alpha001 | 0.01 | First 10 alphabetically | 7, 17, 27 | blend, residual | 60 |
| alpha090 | 0.9 | Same first 10 | 7, 17, 27 | blend, residual | 60 |

These are exactly **720 unique configurations** with no additional seeds,
tickers or repetitions. All three alpha starts are compared on the same first
ten tickers with identical seeds and fusion modes. Sorting uses `LC_ALL=C`.

Each run's output directory follows
`RESULTS/full/{alpha_label}/{fusion_mode}/{ticker}/seed{seed}/output`, with
`train.log` beside `output`. This layout keeps alpha starts and fusion modes
separate while preserving each run's args, history, summary and best checkpoint.

The script resolves CODE from its own location, sets `PYTHONPATH` to CODE and
CODE/forecasting_task, and exports `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2`.
It checks for exactly 100 parquet files and working CUDA before starting training.
It saves the selected ten tickers to `RESULTS/selected_tickers.txt`, the code SHA
to `code_commit.txt`, and a copy of the running script. When Git metadata is not
available inside the container, provide the full SHA without changing Git rights
or configuration:

```bash
CODE_COMMIT=<full-commit-sha> bash run_group_no_weights_720.sh
```

An existing run directory stops the queue without overwriting its contents.
An existing experiment metadata file must match. A failed training process or
missing/empty `summary.json` or `best_checkpoint.pt` stops the queue and prints
the last 30 lines of that run's `train.log`. START and OK report
`completed=N/720`; completion is printed only as `ALL 720 RUNS COMPLETED`.

### Server experiment: 720 runs with linear heads

Run `bash run_group_linear_init_720.sh` inside the existing CUDA-enabled Docker
container. The script resolves the code directory from its own location and sets
`PYTHONPATH` to include that directory and `forecasting_task`. No new dependencies
are installed. Data and output directories are fixed in the script:

```text
DATA=/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts
RESULTS=/auto/datasets/kondyurin/FinTexTS/runs/group_linear_init_720_v1
```

Every model uses five independent `Linear(384, 3)` heads with no hidden layer.
`text_hidden_size=64` is recorded but unused with `text_num_layers=1`.
All configurations use FiLM, sequence length 64, label length 16, horizon 3,
batch size 32, initial alpha 0.1, learning rate 0.001, up to 50 epochs and
early stopping patience 10. Both OMP and MKL thread counts are set to 2.

| Initialization | Weights: macro, sector, targetCompany, relatedCompany, filing | Tickers | Seeds | Fusion modes | Runs |
| --- | --- | --- | --- | --- | --- |
| uniform | 0.2 0.2 0.2 0.2 0.2 | All 100 | 7, 17, 27 | blend, residual | 600 |
| macro | 0.6 0.1 0.1 0.1 0.1 | First 10 alphabetically | 7, 17, 27 | blend, residual | 60 |
| targetCompany | 0.1 0.1 0.6 0.1 0.1 | Same first 10 | 7, 17, 27 | blend, residual | 60 |

The script checks for exactly 100 parquet files and working CUDA before the
queue. Sorting uses `LC_ALL=C`; the ten selected tickers are recorded in
`RESULTS/selected_tickers.txt`. It also saves `code_commit.txt` and an exact copy
of the script. If Git metadata cannot be read inside the container, pass the full
code commit SHA through the environment, for example:

```bash
CODE_COMMIT=<full-commit-sha> bash run_group_linear_init_720.sh
```

Each run writes `train.log` beside its `output` directory under
`RESULTS/full/{initialization}/{fusion_mode}/{ticker}/seed{seed}/`.
Existing run directories are never overwritten; encountering one stops the queue.
Existing experiment metadata must match and is preserved. A failed training
process, or a missing/empty `summary.json` or `best_checkpoint.pt`, stops the queue
and prints the last 30 training log lines. START/OK messages report
`completed=N/720`; only a fully successful queue prints `ALL 720 RUNS COMPLETED`.

### Validation price-anchor diagnostic

Inside the existing Docker, run the independent diagnostic (CPU by default):

```bash
python /workspace/FinTexTS/analyze_blend_price_anchor.py --device cpu
```

The default results/data roots are the same as the fixed-uniform alpha experiment.
`--results_root` and `--data_root` can relocate those inputs. The script first
requires exactly 30 distinct best checkpoints under
`full/alpha090/blend/{ticker}/seed{7,17,27}/output`, with the same ten tickers for
all seeds. It checks the selected ticker list when present. It loads configuration
and trained alpha from each best checkpoint, uses `Dataset_Custom(flag="val")`,
and compares the dataset's train-fitted mean/scale/variance to the saved scaler.
It does not call runner entry points or read saved prediction files.

For the same validation batch it evaluates the trained `full` forward,
`no_text_delta = (1-alpha)*film_output + alpha*last_price`, and persistence
(`last_price` expanded across the horizon). The no-text-delta baseline retains
last price even for absent text. All models run in eval under no-grad; MSE/MAE
are accumulated over all normalized OHLC elements, including the last partial
batch. No inverse transform is applied. Full MSE must reproduce the summary's
`best_val_mse` with `rtol=1e-4, atol=1e-6`; a mismatch stops with the checkpoint
path and both MSE values. No test-split evaluation or training is performed.

After all 30 runs pass, CSVs are written only to
`group_no_weights_alpha_v1/diagnostics/price_anchor/`:

- `runs.csv`: per-run trained alpha, element count, all six errors and comparisons.
- `by_seed.csv`: arithmetic means across ten tickers, percentage changes and win counts.
- `across_seeds.csv`: means and sample standard deviations across three seed means
  (`ddof=1`).
- `comparisons.csv`: overall strict win counts out of 30 and percentage MSE changes
  computed from the overall mean errors.

Percentage change is `100*(full-reference)/reference`; negative values mean full
has lower MSE. A zero reference MSE gives an undefined percentage (blank CSV cell).
Existing diagnostic CSVs are preserved; move a previous report before rerunning.
Original training outputs are never overwritten.

### Persistence comparison across 100 tickers

Inside the existing Docker, run:

```bash
python /workspace/FinTexTS/analyze_persistence_baseline.py --device cpu
```

The script uses the prepared parquet data and the six `full/alpha010` runs per
ticker (blend/residual, seeds 7/17/27) under `group_no_weights_alpha_v1`.
The historical comparison comes from
`film_compare_50ep_v1/full/baseline/{ticker}/seed{seed}/output`.
`--results_root`, `--baseline_root` and `--data_root` can relocate these inputs.
No forecasting models or runner entry points are imported; model errors are
read from summaries, preferring the available `best_val_mse`, `best_test_mse`,
`best_test_mae` and `best_val_mae` fields (or explicitly named split-specific
fields if best fields are absent). Missing metrics remain blank, and each
reported model error records its actual source key.

Exactly 100 distinct ticker parquet files and all 600 primary run configurations
are required. Sequence/prediction/label lengths, four price channels, saved
scalers, dataset preprocessing and declared split parameters are checked before
pooling runs. The current `Dataset_Custom` fixed date policy is recorded explicitly:
data from 2019-01-01 to 2023-12-19 exclusive, training before 2022-01-01,
validation during 2022, and test during 2023 (with the data cutoff applied).
Unknown saved split overrides are rejected rather than silently ignored.

Persistence uses only `seq_price_x[:, -1:, :]`, expanded across `pred_len`;
targets are the last `pred_len` entries of `seq_price_y`. Both val and test are
replayed with `Dataset_Custom`, no shuffling, no dropped final batch, and no
inverse transform. Error sums use all normalized OHLC elements. One val/test
calculation per ticker is reused after the six configurations/scalers match,
and seed/fusion independence of the resulting persistence scores is checked.
No data-loader, forecasting model or training code is modified.

For every primary run, normalized `inputs` and `ground_truths` from `test_io.pt`
are checked against the complete ordered replayed test windows and targets.
Saved-window persistence must agree too. Arrays ending in `_inv` are not used.
Missing normalized arrays are reported with the file's actual keys; primary test
improvements are withheld if saved-window verification is unavailable, and an
actual primary window mismatch stops analysis. Historical baseline comparisons
require matching configuration/scaler/split policy; saved windows are additionally
checked when available. Missing historical saved windows are explicitly labeled,
while configuration/scaler/window mismatches suppress relative improvements.
Available raw historical summary errors remain visible with comparison status.

Reports are written only after successful primary checks, to
`group_no_weights_alpha_v1/diagnostics/persistence_100/`:

- `per_run.csv`: persistence, both new variants and historical baseline, each
  ticker/seed; raw errors, comparability, actual metric keys and test-file checks.
- `per_ticker.csv`: three-seed means, confirmed improvements and stock-level wins.
- `by_seed.csv`: means across all 100 stocks for each seed, with metric availability
  and comparability counts. An incomplete cohort is never substituted for 100 stocks.
- `across_seeds.csv`: mean and sample std (`ddof=1`) between three seed means.
- `wins.csv`: stocks whose three-seed mean test MSE is strictly below persistence,
  with the confirmed denominator and an explicit `all_100_confirmed` indicator.
- `issues.csv`: missing fields/files and reasons for exclusion.
- `configurations.csv`: replay settings, fixed split policy, scaler statistics,
  element counts and hashes of the normalized val/test windows and targets.

`improvement_pct = 100*(persistence_error-model_error)/persistence_error`;
positive means the model is better. Zero-denominator percentages are blank.
Overall improvements require all 100 comparable stocks for each seed. Missing
validation MAE is never filled using test MAE. Original training artifacts and
existing diagnostic CSVs are preserved.

Run synthetic checks without a dataset or GPU:

```powershell
python -m unittest discover -s tests -v
```

These checks cover batch sizes 1 and greater, one/two/deeper MLPs, missing text,
both fusion formulas, gradients and optimizer membership, train-only scaling,
best/final logging, early stopping and exact eval prediction recovery from disk.
They also check weight validation, the three starting distributions, single-layer
head structure and independence of FiLM/head initialization from group weights.
When Bash is available, the suite checks shell syntax and dispatches all 720
configurations of each dispatcher using fake training/CUDA, plus failure and
overwrite guards. Fixed-uniform tests additionally cover the absence of G,
all alpha starts, initialization independence across fusion modes, old checkpoint
migration and exact prediction recovery from disk.

### Context and novelty gate: 60 runs

`--group_gate static|context_novelty` opts into a separate anchored FiLM experiment.
The existing fusion modes retain their behavior. Both conditions use five
independent `Linear(384, pred_len)` heads in order macro, sector, targetCompany,
relatedCompany, filing, and **constant alpha=0.1** (a buffer, no alpha logit or G):

```text
d_g = Linear_g(last_input_day_embedding_g)
mask_g = abs(embedding_g).sum(-1) >= 1e-6
delta = sum_g(w_g * mask_g * d_g)
prediction = 0.9 * FiLM_output + 0.1 * (last_price + delta)
```

The mask applies after each head, including its bias. The correction at each
horizon is shared across the four normalized OHLC channels. Last price is the
last known normalized input OHLC vector, repeated across the prediction horizon;
it remains in the formula with completely missing text. Weights are never
renormalized over present groups.

`static` uses w=0.2 for every group. `context_novelty` forms 18 features:

```text
context = concat(X[-1] - X[-2], X[-1] - X[-6])  # 8 input OHLC features
novelty_g = 1 - cosine(e_current_g, e_previous_present_g)
features = concat(context, novelty_5, pair_available_5)
weights = softmax(Linear(18, 5)(features), dim=-1)
```

The gate has exactly 95 parameters. Weight matrix and bias start at zero, giving
identical initial weights (0.2) and predictions to static. Gate construction
preserves the RNG state; each seed has identical initial FiLM/head parameters
and a separate seeded train-loader generator ensures identical batch permutations.
FiLM, heads and the dynamic gate train jointly with Adam, without new dropout or
regularization. The optimizer includes each trainable parameter once.

`data_provider/context_novelty.py` wraps `Dataset_Custom` only for this experiment.
The base dataset, scaler and split implementation are unchanged. It computes
novelty in chronological row order for one ticker at a time, using the previous
strictly earlier **present** embedding of the same group, including available
history before the current input window and across split boundaries. The current
row is evaluated before updating its history; missing rows do not erase earlier
text. If either embedding is unavailable, novelty and pair_available are zero.
Cosine uses finite float64 normalized vectors, excludes zero norms and clips to
[-1,1]. Embeddings and parquet files are not modified. Features are indexed by
the actual last input row and do not depend on batching or shuffling.

The wrapper verifies one ticker per file, unique chronological dates, and exact
agreement with the base dataset's text/price slices. seq_len must be at least six.
The existing daily row convention is preserved: these data do not identify the
timing of intraday market reactions. Weights are allocation coefficients, not
causal importance or a fraction of news already reflected in prices.

After training, the best validation-MSE checkpoint is restored and validation
runs in eval/no_grad. Its MSE is checked against the selected best value.
`validation_group_diagnostics.csv` has five rows per window, in dataset window
order followed by group order, with:

- window index and actual date of the last input observation;
- group, presence, novelty, pair_available and weight;
- raw head outputs `head_h1..h3`;
- actual normalized prediction corrections `contribution_h1..h3`, equal to
  `0.1 * weight * mask * head` and shared across OHLC channels.

`validation_weights.png` plots weights over time, and
`validation_contributions_h1.png` through `h3.png` plot contributions separately
for each horizon. This diagnostic uses validation only; test metrics never select
the checkpoint. The normal args/history/summary/checkpoint/test_io artifacts and
train/val/test timings remain, with separate `diagnostic_seconds` and total time
including diagnostics. Configurations record fixed alpha, gate structure, novelty
policy, group order and code SHA. Checkpoints include both model states, optimizer,
train-only scaler statistics and train-generator state. Reconstruct text fusion
with `build_text_model(checkpoint['args']['text_model_config'],
checkpoint['group_weight_mode'])`, then load `text_model_state_dict`.

Inside the existing server Docker, from the code checkout:

```bash
bash -n run_context_novelty_gate_60.sh
bash run_context_novelty_gate_60.sh
```

If `.git` is unavailable, set `CODE_COMMIT` to the full deployed code SHA before
running. The dispatcher resolves CODE from its own path, sets PYTHONPATH to include
CODE and CODE/forecasting_task, and uses OMP/MKL threads=2. It checks CUDA and all
eleven required parquet files in
`/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts`.

It first runs NVDA seed=7 for one epoch per condition in `checks/`; these two
runs are separate from the 60 main runs: AAPL, ABBV, ABT, ACN, ADBE, ADI, AMAT,
AMD, AMGN, AMZN × seeds 7/17/27 × static/context_novelty. Main settings are FiLM,
seq_len=64, label_len=16, pred_len=3, batch_size=32, text_num_layers=1,
text_hidden_size=64 (unused), num_epoch=50, patience=10, lr=0.001, alpha=0.1.

Results root is `/auto/datasets/kondyurin/FinTexTS/runs/context_novelty_gate_v1`:

```text
code_commit.txt
selected_tickers.txt
run_context_novelty_gate_60.sh
checks/{condition}/NVDA/seed7/{train.log,output/}
full/{condition}/{ticker}/seed{7,17,27}/{train.log,output/}
```

Existing destinations are rejected before training. Errors stop the queue and
print the log tail; OK requires nonempty summary.json and best_checkpoint.pt.
Progress counts only the main runs, ending in `ALL 60 RUNS COMPLETED`.

`tests/test_context_novelty*.py` covers the formulas, masks, exact zero-gate/static
start, gradients, fixed alpha, initialization and batch permutations, first text,
gaps, cross-split history, future independence, batch independence, base-scaler
agreement, checkpoint/diagnostic reconstruction, and a mocked 2+60 queue with
failure/overwrite checks and Bash syntax validation. No server data or GPU is
needed for these checks.
