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

Group logits `G` start at zero, so `softmax(G)` initially assigns weight 0.2 to
each group. A group is present when its original embedding has absolute sum
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

Run synthetic checks without a dataset or GPU:

```powershell
python -m unittest discover -s tests -v
```

These checks cover batch sizes 1 and greater, one/two/deeper MLPs, missing text,
both fusion formulas, gradients and optimizer membership, train-only scaling,
best/final logging, early stopping and exact eval prediction recovery from disk.
