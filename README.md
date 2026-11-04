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

### Novelty ablation across all 100 tickers

`--group_gate context_no_novelty` adds an ablation of the dynamic gate. Existing
`static` and `context_novelty` behavior is preserved. Both dynamic conditions use
the same zero-initialized `Linear(18,5)` gate (95 parameters):

```text
context_novelty:    concat(price_context_8, novelty_5, pair_available_5)
context_no_novelty: concat(price_context_8, zeros_5,   pair_available_5)
delta = sum_g(weight_g * present_g * head_g(embedding_g))
prediction = 0.9 * FiLM_prediction + 0.1 * (last_price + delta)
```

Only gate input columns 8..12 are zeroed. The original embeddings, five linear
heads, presence masks, price context and pair_available are unchanged. Causal
novelty is still calculated from the same history for both conditions. Validation
CSV diagnostics retain the real `novelty` and add `novelty_used`: real novelty for
context_novelty, zero for context_no_novelty. Existing CSV/plot filenames, window
ordering and best-checkpoint export are retained. The chosen condition is saved
in args, history, summary, checkpoint and text_model_config, so the existing
`build_text_model` reconstruction procedure restores the correct ablation.

Both conditions train FiLM, heads and gate from scratch. Alpha stays fixed at 0.1;
scaling, splits, metrics, early stopping and validation-MSE selection are unchanged.
For one seed both conditions have identical FiLM/head initialization, initial
predictions and train batch permutations. Seeds are used directly, without a
condition-dependent offset.

Inside the existing Docker, from the code checkout:

```bash
bash -n run_context_novelty_100.sh
bash run_context_novelty_100.sh
```

Supply the full deployed code SHA through `CODE_COMMIT` if `.git` is unavailable.
The dispatcher reads exactly 100 parquet files, sorted by filename with LC_ALL=C,
checks ticker uniqueness (also rejecting case-only duplicates), and records all
100 tickers in selected_tickers.txt. Data remain under
`/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts`.

Two NVDA seed=7 one-epoch checks run first, followed by **600 new main runs**:
100 tickers × seeds 7/17/27 × context_novelty/context_no_novelty. Main settings are
FiLM, seq_len=64, label_len=16, pred_len=3, batch_size=32, num_epoch=50, patience=10,
Adam lr=0.001, all five groups, text_num_layers=1 and fixed alpha=0.1.
Results use the new root
`/auto/datasets/kondyurin/FinTexTS/runs/context_novelty_100_v1`:

```text
checks/{condition}/NVDA/seed7/{train.log,output/}
full/{condition}/{ticker}/seed{7,17,27}/{train.log,output/}
```

The previous 10-ticker results are neither reused nor moved. The dispatcher
preserves CUDA preflight, CODE/PYTHONPATH resolution, OMP/MKL threads=2, SHA and
script-copy recording, destination overwrite guards, log tails on failure, and
nonempty summary/checkpoint checks. START/OK progress excludes the two checks and
ends at N/600 with `ALL 600 RUNS COMPLETED`.

Synthetic checks verify that only novelty inputs are ablated, those five gate
weight columns receive exactly zero gradients in the ablation and can receive
nonzero gradients with novelty, initial states/predictions and batch order match,
both checkpoints and diagnostics restore correctly, and the mocked dispatcher
contains exactly 600 unique combinations plus two checks in the prescribed order.

### Price-context-only gate: isolated pair-availability ablation

`--group_gate context_price_only` keeps `Linear(18,5)` and its zero initialization,
but passes `concat(price_context_8, zeros_like(novelty), zeros_like(pair_available))`
to the gate. Relative to context_no_novelty, only the five pair-availability inputs
are removed. The embeddings, independent linear heads and mask after each head
retain their behavior. Alpha is fixed at 0.1 and fusion remains anchored blend.
Initial FiLM/head parameters, initial forecasts and train batch order match the
other conditions at the same seed.

Validation diagnostics keep real `novelty` and `pair_available` alongside
`novelty_used` and the new `pair_available_used`. Both used fields are zero in
context_price_only. Existing conditions retain their actual gate inputs. The CSV
and plots are exported from the best validation checkpoint; saved configuration
selects the condition during reconstruction.

Inside the existing Docker, from the code checkout:

```bash
bash -n run_context_price_only_30.sh
bash run_context_price_only_30.sh
```

The dispatcher first runs a separate NVDA seed=7 one-epoch check. It then trains
30 new runs for AAPL, ABBV, ABT, ACN, ADBE, ADI, AMAT, AMD, AMGN, AMZN × seeds
7/17/27. Settings remain FiLM, seq_len=64, label_len=16, pred_len=3, batch_size=32,
five Linear(384,3) heads, alpha=0.1, Adam lr=0.001, max epochs=50, patience=10.
The train-only scaler, temporal splits, missing-text handling and metrics are
unchanged. CUDA preflight, log tails, artifact checks, overwrite guards, SHA/script
recording and CODE_COMMIT fallback are preserved. Results are separate:

```text
/auto/datasets/kondyurin/FinTexTS/runs/context_price_only_10_v1/
  checks/context_price_only/NVDA/seed7/{train.log,output/}
  full/context_price_only/{ticker}/seed{7,17,27}/{train.log,output/}
```

After the server runs finish, compare their saved metrics on CPU:

```bash
python analyze_context_price_only.py
```

The script reads the same ten tickers and three seeds from the new runs,
`context_novelty_100_v1/full/context_no_novelty/`, and
`film_compare_50ep_v1/full/baseline/`. No inference model or training entry point
is loaded. It checks saved FiLM/window/training settings, ticker/seed identity,
checkpoint selection epoch, split policy and saved OHLC scaler agreement. The
two controlled variants must also match novelty/sampling policies and head/gate
structures. A primary mismatch stops analysis; an unavailable or incompatible
historical baseline is labeled with a reason and excluded from paired comparisons
and selection. Available raw baseline errors remain visible. This is a saved
configuration/scaler comparison; it does not replay parquet data or verify saved
test windows. Missing metrics are never replaced by another split's errors.

`--results_root`, `--reference_root`, `--baseline_root`, and `--output_dir` can
relocate inputs/reports. Reports default to the new results root's
`diagnostics/comparison_10/`, and an existing report directory is rejected:

- `per_run.csv`: validation MSE, test MSE/MAE, actual source fields and comparability.
- `per_ticker.csv`: each stock's three-seed averages.
- `by_seed.csv`: each seed's arithmetic means over all ten stocks.
- `across_seeds.csv`: mean and sample std (ddof=1) between the three seed means.
- `paired_by_ticker.csv`: paired differences after averaging seeds; left minus
  right, negative means the left variant has lower error.
- `wins.csv`: strict wins and ties for each pair/metric after averaging seeds,
  with the confirmed denominator out of ten.
- `issues.csv`: missing metrics and baseline comparability failures.
- `selection.json`: variant(s) minimizing mean validation MSE; test errors never
  participate in selection. Averages require complete cohorts rather than subsets.

Seed/statistics tables, paired stock differences, win counts and selection are
also printed. Tests cover pair-availability invariance with a nonzero gate,
presence masking, zero gradients in the ten unused columns, matched initialization
and batching, checkpoint/diagnostic restoration, a mocked 1+30 queue, statistical
aggregation, mismatch exclusions, read-only analysis and validation-only selection.

### Price-only gate with a five-row text mean

`--text_temporal_pool last|mean5` defaults to `last`, preserving the previous
last-input-row behavior. `mean5` is an isolated option for `context_price_only`.
For each group, it considers exactly the last five **dataset rows** of the current
input window (not five calendar days or the last five nonempty dates). Daily
presence is computed before pooling with float32 `abs(vector).sum(-1) >= 1e-6`.
Only present vectors enter the mean; a completely empty group gets a zero vector.
There is no extra normalization, forward/backward fill or history search.
Forward-fill dataset paths are rejected for mean5.

Group presence is `available_dates > 0`, independently of the mean's norm.
Opposite nonzero vectors can cancel to zero and still leave the group present.
The independent Linear(384,3) head receives the mean and its output, including
bias, is masked **after** Linear. The gate receives only the same eight price
context features plus ten zeros. Neither counts nor window presence enter it.

```mermaid
flowchart TD
    P["Known normalized OHLC window"] --> F["FiLM: price prediction"]
    P --> C["Price context: X_last-X_prev, X_last-X_5rows_earlier"]
    C --> G["Linear(18,5) + softmax: context_8 and zeros_10"]
    T["Separate text groups in input window"] --> W["Last five dataset rows per group"]
    W --> M["Daily float32 L1 presence masks"]
    W --> A["Mean of present vectors per group"]
    M --> A
    M --> N["Count and present_in_window = count > 0"]
    A --> H["Five independent Linear(384,3) heads"]
    H --> D["Mask after heads; weighted sum of group corrections"]
    N --> D
    G --> D
    P --> L["Last known normalized OHLC"]
    F --> O["0.9 price + 0.1 last OHLC + 0.1 correction"]
    L --> O
    D --> O
```

The OHLC correction at each horizon is shared across channels. Alpha remains a
fixed 0.1 buffer. Gate matrix/bias still start at zero and weights at 0.2.
Pooling consumes no random numbers: at one seed FiLM/head initialization and train
batch order match last. All other settings remain FiLM, groups macro/sector/
targetCompany/relatedCompany/filing, seq_len=64, label_len=16, pred_len=3,
batch_size=32, Adam lr=0.001, max epochs=50, patience=10, with the same train-only
scaler, temporal splits, metrics and validation-MSE checkpoint selection.

Inside the existing Docker, from the checkout:

```bash
bash -n run_context_price_mean5_30.sh
bash run_context_price_mean5_30.sh
```

The script first runs NVDA seed7 for one epoch, then 30 main runs on AAPL, ABBV,
ABT, ACN, ADBE, ADI, AMAT, AMD, AMGN, AMZN × seeds 7/17/27. DATA stays
`/auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts`. The new root is
`/auto/datasets/kondyurin/FinTexTS/runs/context_price_mean5_10_v1`, with
`checks/context_price_only/NVDA/seed7/output` and
`full/context_price_only/{ticker}/seed{seed}/output`, and adjacent train.log files.
CUDA preflight, existing-directory rejection, artifact checks, failure log tails,
SHA/script copies, CODE_COMMIT fallback and N/30 progress are preserved.

args.yaml, history.jsonl, checkpoint and summary record text_temporal_pool.
Validation CSV/plots still come from the best checkpoint. CSV rows now also
record inclusive `text_window_start_date`, `text_window_end_date`,
`available_dates` per group and the actual `present` mask. Dates come from real
dataset row indices. New mean5 runs save `window_evidence.json`, also embedded
in checkpoint args and summary: window counts/shapes and float32 SHA-256 hashes
of normalized val/test inputs, future targets, input-end dates and target dates.
Old checkpoint configs without a pooling field restore as last.

After training, run the comparison on CPU:

```bash
python analyze_context_price_mean5.py
```

The primary comparison is mean5 versus last from `context_price_only_10_v1`.
The same stocks/seeds from `context_novelty_100_v1/full/context_no_novelty` and
`film_compare_50ep_v1/full/baseline` are additional references. Saved settings,
scalers, split policy and head/gate structures are checked. Val/test windows are
replayed with Dataset_Custom and their normalized targets, inputs and dates are
fingerprinted. New mean5 saved fingerprints must match; normalized `inputs` and
`ground_truths` in each test_io.pt must match the replay exactly. Inverse arrays
are never used. Saved validation diagnostic dates/order must agree when available
(required for the controlled variants). Legacy runs lack saved validation target
tensors: their validation evidence is the replayed configuration/targets and saved
input-end dates, rather than a historical target-tensor hash. Per-run
`window_verification` records the available evidence. A primary mismatch stops the
report; incompatible or unavailable extra references are explicitly excluded from
paired comparisons while available raw metrics remain visible.

Reports default to the new root's `diagnostics/comparison_10/`. CLI overrides are
`--results_root`, `--last_root`, `--reference_root`, `--baseline_root`, `--data_root`,
and `--output_dir`. Existing report directories and training results are preserved.
Outputs include per_run/per_ticker, by_seed and across_seeds tables for validation
MSE and test MSE/MAE; sample std is between three seed means with ddof=1.
paired_by_ticker and wins compare all pairs after averaging seeds. Negative
left-minus-right and relative changes mean lower error for the left variant.
relative_by_seed contains seed-relative changes, their mean/sample std, and the
relative change of the overall means. target_windows records replayed signatures,
and issues records exclusions. selection.json chooses only between mean5 and last
using mean validation MSE; test scores and the two references do not select pooling.

**Mandatory next stage:** compare mean5 with a repetition-aware aggregation of
the **same five dataset rows**, holding the price-only gate, head architecture,
alpha, split policy and training budget fixed. That second mechanism is not
implemented in this experiment.
