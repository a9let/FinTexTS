# Text history in the group gate only

Branch: `experiment/context-price-gate-history`, from
`experiment/context-price-uniform-history`.

`--gate_text_history_mode none|history|difference|uniform_history` defaults to
`none`. Enabled modes require FiLM, `context_price_only`,
`text_temporal_pool=last` and `text_history_mode=last`. The original five
Linear(384,3) prediction heads receive only the current embedding; no additional
prediction heads or attention pool are constructed.

For each group, reuse the existing current-versus-previous-four-row calculation:

```text
K_j = exp(-(1-cosine(e_t,e_j))/0.1), excluding missing history
m = sum_j K_j e_j / sum_j K_j; rho = max_j K_j
history:         z_g = m
difference:      z_g = e_t - rho*m
uniform_history: z_g = mean of available previous four original embeddings
price_logit_g = original_price_gate(price_context)[g]
text_logit_g = dot(u_g, z_g)
weight_g = softmax_groups(price_logit + text_logit)[g]
```

The four previous rows are within the input window, separately per group; the
current row never enters m. Presence is the existing float32 L1 >= 1e-6 rule.
No history gives m=0 and rho=0; difference then receives e_t. Missing current text
forces z and its logit addition to zero, even with available history. The original
post-head mask remains. Softmax still includes all five groups, with no
renormalization over present text. The price gate uses the original lags [1,5];
novelty_used and pair_available_used are zero.

`gate_text_vectors` contains five independent rows of length 384 (1,920 trainable
parameters), initialized to zero without RNG consumption. FiLM, head/gate initial
parameters, predictions and training permutations match the control. With `none`,
no new parameter is saved and the original softmax path remains unchanged, also
for restored nonzero price-gate weights. All existing experiment modes remain
available when this option is none.

Fusion stays `0.9*FiLM + 0.1*(last_price + masked_group_delta)`. Alpha is fixed,
and FiLM, scalar normalization, splits, metrics, Adam, early stopping and timing
remain on the existing runner path. Args/checkpoint/summary save the mode and
gate-vector configuration; state dictionaries restore the new vectors. History
logs include the enabled gate mode. Best-validation diagnostics add `price_logit`,
`gate_text_logit`, `total_logit` and the existing `weight`, separately per group.
These are arithmetic gate quantities, not causal news evidence.

## Server execution inside the existing Docker

From the checkout:

```bash
bash -n run_context_price_gate_history_90.sh
bash run_context_price_gate_history_90.sh
```

The dispatcher first verifies the 30 saved controls under
`runs/context_price_only_10_v1/full/context_price_only/`. `gate_history_control.py`
checks settings, original gate/head structure, ticker/seed, checkpoint selection,
train-only scaler, archived validation dates, normalized saved test inputs/targets
and replayed best validation MSE (rtol=1e-4, atol=1e-6). Only validation inference
is performed in the preflight; test data are used only to verify saved windows
and targets. Control parameters/results are never modified or used to initialize
the new training models.

Old controls lack historical validation tensor hashes. Such verification is
explicitly labeled `partial_historical_validation`: replayed best MSE and saved
dates agree, but historical validation tensor identity is not cryptographically
proven. Historical embedding values are not hashed either. Any available saved
fingerprint contradiction or MSE/config/scaler/window mismatch stops the queue.

The preflight writes a new `control_verification.json` and log. Every main run
receives `--gate_control_manifest`; before constructing its models it checks its
settings, scalers and validation/test input/target/date fingerprints against the
verified replay. The selected reference and source checkpoint hash are copied
into args/checkpoint/summary. Manual runs can use the same manifest. NVDA checks
are outside the control cohort and do not receive a reference manifest.

There are three NVDA seed=7 one-epoch checks, followed by exactly 90 fresh main
runs: history/difference/uniform_history x AAPL ABBV ABT ACN ADBE ADI AMAT AMD
AMGN AMZN x seeds 7/17/27. Each main run uses seq_len=64, label_len=16, pred_len=3,
batch_size=32, max epochs=50, patience=10 and Adam lr=0.001. Both text modes are
last in all commands. The saved control is never retrained.

Results are separate:

```text
/auto/datasets/kondyurin/FinTexTS/runs/context_price_gate_history_10_v1/
  checks/{history,difference,uniform_history}/NVDA/seed7/{train.log,output/}
  full/{history,difference,uniform_history}/{ticker}/seed{7,17,27}/{train.log,output/}
  control_verification.json
  control_verification.log
```

The dispatcher retains CUDA checks, OMP/MKL threads=2, CODE/PYTHONPATH discovery,
SHA/script recording, CODE_COMMIT fallback, overwrite guards, stop-on-error log
tails, and nonempty summary/checkpoint checks. Progress excludes checks and ends
with `ALL 90 RUNS COMPLETED`. Reports and existing results are never overwritten.

Synthetic CPU tests cover legacy none behavior, initial equality and RNG/batching,
explicit logit formulas, missing/no-history cases, group independence and causal
windows, gradients and optimizer membership, checkpoint/diagnostic restoration,
control replay/binding failures, and a mocked 3-check + 90-main queue. No real
training, server replay or push is part of local verification.
