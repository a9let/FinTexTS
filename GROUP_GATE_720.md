# HAN / Huang / T-LSTM: price_only против uniform

Ветка `experiment/group-gate-ablation-720` подготовлена в отдельном worktree,
без изменения пользовательских рабочих копий, run_baselines.sh или результатов.

## Что перенесено и что добавлено

- Основа — `experiment/context-price-huang-replace-grid`, `e642a6d`.
  `GroupHuangQuery.py` не изменён: используется именно `huang_recurrent_replace`,
  пять отдельных LSTM, без параллельной головы last. Проекция текста/запрос 32D,
  состояние H=16/32, шаги t−4…t либо t−19…t, история каждого шага s−10…s−1.
- **Дословно перенесены** `GroupHANTemporal.py` и `HANHeadAblation.py` из
  `experiment/context-price-han-window`, `6a62808`. Полные SHA и SHA-256 файлов
  сохранены в `experiment_group_gate_sources.json`, тест проверяет совпадение.
  Прежние проекция 384→32, BiGRU отдельно для каждой группы, конкатенация состояний,
  HAN attention и Linear(2H→3) сохранены. Нет моста 384D, MLP или ветки last.
  Для H=32/W=5 меняется только window_size прежнего pool, theta берутся 15…19.
- Добавлена T-LSTM по спецификации задания:
  [Baytas et al., KDD 2017, §3.1.2](https://biometrics.cse.msu.edu/Publications/MachineLearning/Baytasetal_PatientSubtypingViaTimeAwareLSTMNetworks.pdf).
  Пять независимых ячеек LSTMCell(384,H), пять W_d/b_d и Linear(H→3).
  На входе исходные MiniLM-векторы, без усреднения дат или дообучения эмбеддингов.

Для доступного текста T-LSTM изменяет только память c, сохраняя h перед вентилями:

```text
short = tanh(W_d @ c + b_d)
c_adjust = c - short + short / ln(e + delta_days)
(h,c) = LSTMCell(x, (h,c_adjust))
```

Delta — календарные дни между доступными датами той же группы внутри окна:
пятница→понедельник=3. Первый текст не затухает. Пропуски не вызывают ячейку,
не обновляют h/c или предыдущую дату. После последнего текста дополнительного
затухания нет. Все окна начинают с h=c=0. Нет attention, novelty или last.
Обычной LSTM в очереди нет: эффект самого затухания относительно неё не изолируется.

## Объединение и воспроизводимость

```text
price_only: softmax прежнего Linear18→5, контекст OHLC[-1]-OHLC[-2/-6]
uniform:    фиксированные 0.2, без обучаемого gate
contribution[g,h] = 0.1 * weight[g] * present[g] * head[g,h]
output[h,c] = 0.9*FiLM[h,c] + 0.1*last_OHLC[c] + sum_g contribution[g,h]
```

Наличие — float32 L1≥1e−6 до преобразований. Маска применяется после головы,
включая bias; веса оставшихся групп не перенормируются. Поправка горизонта общая
для четырёх OHLC. Пять групп в прежнем порядке. Uniform удаляет gate после
инициализации энкодера, сохраняя общие параметры, RNG и порядок батчей пары.
Фиксированные коэффициенты записываются как fixed. Начальные прогнозы пары совпадают.
Головы Huang остаются нулевыми, HAN — прежними; новые головы T-LSTM используют
стандартную torch-инициализацию. Между разными архитектурами равенство не требуется.

FiLM, 64 цены, label16/pred3, batch32, Adam lr=0.001, max50/patience10, loss,
train-only scaler, разбиения и выбор checkpoint по validation MSE не меняются.
Dataset_Custom не меняется. Его обёртка опционально добавляет calendar_days
из дат текущего входного окна; эмбеддинги не преобразуются.

## CLI и очередь

```bash
export PYTHONPATH="$PWD:$PWD/forecasting_task${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u -m forecasting_task.run --model_type FiLM \
  --text_family tlstm --family_hidden_size 16 --text_window 5 --group_weight_condition uniform \
  --root_path /auto/datasets/kondyurin/FinTexTS/minilm_reproduction_v1/fintexts \
  --data_path AAPL.parquet --seq_len 64 --label_len 16 --pred_len 3 --batch_size 32 \
  --text_num_layers 1 --text_weight 0.1 --e_layers 2 --lr 0.001 \
  --num_epoch 50 --patience 10 --seed 7 \
  --logdir /auto/datasets/kondyurin/FinTexTS/runs/group_gate_single_v1/output
```

Семейства han/huang/tlstm, H=16/32, окно/шаги=5/20, price_only/uniform.
Все 12 конфигураций обучаются в двух режимах: 24×10 тикеров×3 seeds=720.
Тикеры AAPL ABBV ABT ACN ADBE ADI AMAT AMD AMGN AMZN; seeds 7,17,27.
Перед ними — 24 одноэпоховых NVDA seed7 smoke check, вне счётчика 720.

`run_group_gate_ablation_720.sh` проверяет CUDA, использует полный SHA через
Git/CODE_COMMIT, OMP/MKL=2, CODE/PYTHONPATH по месту скрипта. Сохраняет копии
диспетчера, args, history, модель/optimizer/best checkpoint, summary и время.
Артефакты содержат конфигурацию и отпечатки данных/окон/целей/дат, начальных
параметров и фактического порядка батчей. Возобновление пропускает только
подтверждённые совпадающие результаты; неполные/несовпадающие каталоги сохраняются
и останавливают очередь. Ошибка обучения выводит хвост train.log.
Лог: START/OK/SKIP, N/720, ETA по семействам, ALL 720 RUNS COMPLETED.
12 часов — ориентир, не тайм-аут и не гарантия; нужен средний прогон ≤1 минуты.

```text
/auto/datasets/kondyurin/FinTexTS/runs/group_gate_ablation_720_v1/
  checks/{family}_h{H}_w{W}_{weights}/NVDA/seed7/{train.log,completed.json,output/}
  full/{family}_h{H}_w{W}_{weights}/{ticker}/seed{seed}/{train.log,completed.json,output/}
  controls/full/last_{weights}/{ticker}/seed{seed}/output/
  diagnostics/gate_comparison/
```

## Контроли и анализ

Сохранённые last_price_only и last_uniform берутся соответственно из
`context_price_han_head_ablation_10_v1/full/last_matched` и
`context_novelty_gate_v1/full/static`. Флаги `--last_price`/`--last_uniform`
переопределяют эти каталоги над тикерами. Проверяются данные, нормализованные
ценовые окна/цели/даты, scaler, формула anchored blend, маски, настройки,
инициализация FiLM и доступные доказательства порядка батчей.
Неполная историческая сверка — partial, без строгих процентов/выводов.
Автоматического переобучения нет. `control_rerun_commands.txt` содержит команды
только для неподтверждённых контролей:

```bash
bash run_group_gate_ablation_720.sh --controls price_only
bash run_group_gate_ablation_720.sh --controls uniform
```

Каждый такой явный запуск — 1 check + 30 контролей; `--controls all` — 2+60.
Они не входят в 720 и берут только последний эмбеддинг. Новые подтверждённые
контроли при следующем анализе имеют приоритет. Для повторного отчёта нужен
новый `--output`, существующие файлы не перезаписываются.

Анализатор по умолчанию печатает один короткий русский блок без CSV/JSON и тикеров.
Выбор — только по validation; расхождения победителей validation/test отмечаются.
Если никто из подтверждённых вариантов не обошёл last, это написано прямо.

- `metrics.csv`: 24 варианта + 2 контроля, val MSE, test MSE/MAE, sample std
  между тремя seed-средними и метрики каждого горизонта. Новые val-горизонты
  вычисляются на полном проходе лучшего checkpoint, test — по нормализованным
  outputs/targets. Недоступные исторические поля пустые.
- `gate_comparison.csv`: ровно 12 пар price_only vs uniform, разницы,
  100*(ошибка uniform−ошибка price_only)/ошибка uniform, победы по акциям после
  среднего по seeds, число подтверждённых пар и статус.
- `architecture_comparison.csv`: 24 сравнения с last того же режима весов.
- `contributions.csv`: подписанные вклады групп/горизонтов, веса и маски
  лучшего validation-checkpoint. Перед экспортом проверяется формула вклада
  и восстановление каждого OHLC-прогноза. Веса дат LSTM не объявляются
  аддитивным разложением прогноза.
- `prediction_components.csv`: FiLM, anchor, text и прогноз по каждому
  каналу/горизонту; соединяется с contributions по variant/ticker/seed/window_index
  для независимой проверки восстановления без изменения обучающих файлов.
- `per_run.csv`, `by_seed.csv`, `verification.csv`, `paired_by_ticker.csv`:
  подробная сверка и парные результаты. Плюс процента означает снижение ошибки.

## astra

После отдельного коммита и публикации ветки, из host Git checkout:

```bash
set -euo pipefail
git fetch origin
VARIANTS_HOST=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace/variants"}}{{.Source}}{{end}}{{end}}' kondyurin-fintexts-compare)
test -n "$VARIANTS_HOST"
WORKTREE="$VARIANTS_HOST/group-gate-ablation-720"
git worktree add --detach "$WORKTREE" origin/experiment/group-gate-ablation-720
CODE_COMMIT=$(git -C "$WORKTREE" rev-parse HEAD)
CONTAINER_CODE=/workspace/variants/group-gate-ablation-720
docker exec kondyurin-fintexts-compare python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
docker exec -d -e CODE_COMMIT="$CODE_COMMIT" -w "$CONTAINER_CODE" \
  kondyurin-fintexts-compare bash -lc '
    set -euo pipefail
    mkdir -p /auto/datasets/kondyurin/FinTexTS/runs/group_gate_ablation_720_v1
    exec bash run_group_gate_ablation_720.sh >> /auto/datasets/kondyurin/FinTexTS/runs/group_gate_ablation_720_v1/queue.log 2>&1
  '
docker exec kondyurin-fintexts-compare tail -n 60 -f \
  /auto/datasets/kondyurin/FinTexTS/runs/group_gate_ablation_720_v1/queue.log
# После очереди:
docker exec -w "$CONTAINER_CODE" kondyurin-fintexts-compare python analyze_group_gate_ablation.py
```
