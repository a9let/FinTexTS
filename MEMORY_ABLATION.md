# MEMTO, Titans, xTime. Group weights ablation

Branch `experiment/group-memory-ablation-720` starts at
`2fbe38cc599e942e1a1d932bb1cb71ecbdf1cc7a`. The OPT-in flags are
`--text_family memto_group|titans_linear_group|xtime_group`,
`--text_window 5|20`, and `--group_weight_condition price_only|uniform`.
For MEMTO, `--family_hidden_size 8|16` denotes K. For Titans it denotes H,
16 or 32. xTime uses the default size placeholder 32 and `--kd_beta 0|0.5`.

The base runner uses FiLM, normalized 64-row OHLC input, label length 16,
three horizons, batch size 32, Adam at 0.001, 50 epochs and patience 10.
The scaler fits train only. The usable data span starts on 2019-01-01 and
ends before 2023-12-19. Train targets end before 2022-01-01, validation targets
are in 2022, and test targets are in the 2023 portion of that data span.
The base dataset overlaps preceding input rows at split boundaries. This
overlap, normalized OHLC targets, window order, loss and ordinary metrics
follow the base runner. The older families retain their configurations.

Five independent text groups use a suffix of W input rows. Presence is float32
L1 >= 1e-6 before any projection. Group outputs are masked after the heads.
Dynamic weights use the existing normalized OHLC lags 1 and 5. Fixed weights
are 0.2 even when groups are missing. Both use the same anchored alpha=0.1
blend. xTime applies expert weights to group deltas before this blend.

## Research adaptations

- [MEMTO §3](https://arxiv.org/html/2312.02530#S3) supplies gated prototype
  writes, reads, K-means initialization, and the latent/input deviations.
  Here each group uses one causal masked attention block (width 32, one head,
  residual and LayerNorm) and a single linear 64→384 reconstruction layer.
  With tau=1, writes normalize across available dates and reads across
  prototypes. During training each window derives a differentiable candidate
  from the same pre-batch bank, and reconstructs using its own candidate.
  After backpropagation, the shared bank takes the mean of nonempty candidates.
  An empty group leaves its bank alone. Train budgets are 5 random-bank and
  10 post-clustering epochs. Both minimize reconstruction MSE plus 0.01 times
  the read entropy over present rows. K-means uses all available train queries,
  seed-specific initialization and n_init=10. Too few distinct queries for K
  fail explicitly. The encoder, decoder, bank and train-only feature statistics
  are frozen for the supervised stage. The shared preparation key omits group
  weighting. LSD uses the minimum squared Euclidean distance, following the
  nearest prototype and squared norm in §3.4, Eq. (9). ISD is the squared
  reconstruction norm. A is computed over the present dates at the forecast
  origin. These are retrospective within-window text scores. The 68-feature
  group head sees q/read from the last available row plus four standardized
  scalars. The scores support diagnostics of unusual text.
- [Titans §3 and Appendix C](https://arxiv.org/html/2501.00663#S3) motivates
  the linear association update. Each group projects keys, values and queries,
  L2-normalizes keys/queries with eps=1e-8 and initializes M0 to learnable zeros.
  Sigmoid coefficient weights start at zero and biases at logit(0.05,0.5,0.1)
  for forgetting, momentum and write strength. Per-window memory uses the
  analytic gradient with an uninterrupted differentiable recurrence, and
  reads the last available query after its write. The experiment tests that
  mechanism with the base FiLM and has the same supervised budget.
- [xTime §IV](https://arxiv.org/html/2510.20651#S4) motivates hierarchical
  expert transfer and routing. Each expert has five independent Linear(384→3)
  heads on the masked within-window mean. Normal learns on all train windows
  with the original OHLC-MSE loop, including FiLM and the price gate when present.
  The best validation checkpoint and normal-only metrics are shared by both
  beta variants for a given ticker, seed, W and weighting. FiLM and gate then
  freeze. Moderate and extreme each train for 10 epochs on their own class.
  Each initializes from the preceding expert. Loss is OHLC-MSE plus 0.1 times
  the stated two-sided tail penalty and beta times bounded KD. The tail
  penalty averages over all horizons, with zeros outside the threshold,
  and uses train return population std with eps=1e-8. This is our two-sided
  objective. Router training uses train CE for five epochs, with a shared
  36→16→3 MLP over three forecasts. Inference does a normalized top-2 and uses
  no true rarity label. All heads and the router freeze for final evaluation.
  Normal and rare budgets, router time, and normal-only metrics are recorded
  separately. The final xTime runner epoch measures a frozen model.

Rarity is max absolute raw-Close return over the three horizons. Percentiles
90 and 97 use complete train windows only. Normal includes p90, moderate
includes p97, and rare is moderate plus extreme. Reported MSE for every
stratum still uses normalized OHLC. Empty evaluation strata have no metric.
Rare window counts are window-seed observations. Repeated seeds do not add
distinct market dates.

## Queue and reports

After placing the branch in the existing Docker worktree, set CODE_COMMIT to
its full Git SHA if Git is unavailable to the container. Run from its root.

```bash
bash run_group_memory_ablation_720.sh --stage smoke
bash run_group_memory_ablation_720.sh --stage pilot --ticker AAPL --seed 7
bash run_group_memory_ablation_720.sh --family memto_group
bash run_group_memory_ablation_720.sh --family titans_linear_group
bash run_group_memory_ablation_720.sh --family xtime_group
bash run_group_memory_ablation_720.sh
bash run_group_memory_ablation_720.sh --stage control
python analyze_group_memory_ablation.py
```

Smoke uses all 24 cells on NVDA seed 7 with one epoch per stage. Pilot uses
the full protocol and the same canonical paths as the full queue. The result
root is `/auto/datasets/kondyurin/FinTexTS/runs/group_memory_ablation_720_v1`.
The 30 matched uniform controls are under controls/full, outside the 720.
The stage logs, resume snapshots, configuration manifest and validation
artifacts stay under this root. Resume keeps prior attempt logs and checks
the configuration and source fingerprints. The dispatcher serializes
execution for a result root. ETA uses measured runs and epoch progress.
Actual GPU time needs smoke and pilot measurements.

The analyzer rechecks the saved last_price_only against the same inputs and
protocol, reads matched last_uniform separately, and copies the previous
architecture summary as historical context when it exists. Incomplete pairs
have no strict improvement percentage. All configurations are ranked only
by validation MSE. Reports contain ordinary/rare errors, horizon metrics,
seed spread, paired comparisons, and method diagnostics from selected
checkpoints. A small number of rare windows limits the inference.
Use a new `--output` directory for a later report.
