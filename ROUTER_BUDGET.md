# Проверка бюджета router

Ветка experiment/context-price-mean-router-budget создана от фактического
коммита df0fdd20a9722935fa73489cb807db9f877c7d86 ветки
experiment/context-price-mean-router. Отличий базы от указанной серверной
версии нет. Отдельный worktree находится вне .venv.

Эксперимент использует полные серверные cache сохранённых прогнозов.
Исходный каталог доступен коду только для чтения. Новый entry point
router_budget.py предоставляет plan / audit / train / analyze / queue.
Обёртка run_context_price_mean_router_budget.sh запускает audit → train →
analyze. Ни одна стадия не вызывает create_cache, обучение или forward
экспертов. CUDA не требуется.

Очередь содержит ровно девять новых fit: constant / price / price_text ×
seeds 7/17/27. У каждого своя нулевая инициализация. Новые пределы явно
заданы --max-epochs 1000 --patience 100. AdamW lr=.001, weight_decay=0,
регуляризация .001×mean(theta²), чанки 128 и один optimizer step на эпоху
сохранены. Среднее по акциям имеет равный вес при разных числах окон.
Градиент MSE накапливается по всем чанкам, L2 добавляется один раз.

Сохраняются Router, признаки и схема, epsilon, scaler признаков по Q2–Q3,
порядок 15 полных экспертов и пяти групп. Q4 MSE без регуляризации выбирает
checkpoint. Q1 не используется для обучения router. Полный validation-2022
в отчёте является описательным.

## Initial, checkpoint и resume

Новая явная опция fit_router(include_initial=True) включает шаг 0 среди
кандидатов. Значения по умолчанию прежнего режима остаются 200/20 без initial.
Новый entry point всегда включает initial в signature.

До первого шага сохраняются initial_checkpoint.pt, best_checkpoint.pt и
last_checkpoint.pt с epoch=optimizer_steps=0, пустым optimizer и wait=0.
Проверяется совпадение прогноза с equal_mixture при rtol=1e-5, atol=1e-6.
Строгое улучшение Q4 сохраняет новый best, поэтому ties оставляют более
ранний checkpoint. Возвращаемая модель нового fit загружает best.

История содержит отдельную строку 0. epochs_completed и optimizer_steps
считают только реальные обновления. Resume разрешён через --resume для
нового fit с идентичной signature. Восстанавливаются optimizer, RNG,
generator и patience; scheduler сохраняется как None. Завершённые fit
пропускаются после проверки digest всех артефактов. Частичные и
несовместимые результаты сохраняются и вызывают ошибку.

## Аудит и provenance

--source-root по умолчанию указывает на
/auto/datasets/kondyurin/FinTexTS/runs/context_price_mean_router_q1_v1/FiLM.
--output указывает на отдельный
/auto/datasets/kondyurin/FinTexTS/runs/context_price_mean_router_budget_v1/FiLM.
Вложение output в source и обратное вложение отклоняются. Lock, логи,
checkpoint, временные файлы и отчёты находятся в новом output.

Audit проверяет manifest с ровно 450 экспертами и 30 baseline, канонические
пути, digest best/last/args/history/summary, Q1-выбор, архитектуру сохранённых
текстовых и финансовых частей, source SHA, scaler и data_audit.
Затем проверяются cache hashes, schema, seed/backbone, порядок экспертов
из закреплённого исходного кода и связанного manifest, группы, размеры,
даты, нормализованные цели и Q1-ошибки по каждой позиции оси экспертов.
У старого cache отсутствует отдельное поле expert_order; основание
восстановленного порядка явно записывается в source_audit.json.
Проверяются исходные router checkpoint, их 200/20 signatures и scaler.

Source code SHA и implementation SHA записываются отдельно. Новый SHA
не подменяет старый в signatures источников. Отсутствующие или
несовместимые артефакты вызывают ошибку. Генерации cache при этом нет.
Train открывает только validation cache. Analyze фиксирует все Q4-выборы
в selection.json перед первым чтением test cache.

Статус verified_saved_artifacts относится к сохранённым свидетельствам.
Raw-data replay и forward экспертов здесь не выполняются. Ограничения
исторической сверки из data_audit сохраняются. Локально используются
синтетические тесты; проверка настоящего серверного пула ещё не выполнена.

После новых fit сверяются train/Q4 MSE на общих шагах со старой историей.
old_new_trajectory.json указывает длину пересечения и наличие всех старых
шагов. Если новый early stopping произошёл раньше, отсутствующий хвост
не объявляется проверенным. Численное сравнение градиентов и параметров
с исходным fit из df0fdd2 выполняется в синтетическом regression test.

## Отчёт

Отчёт в reports/final/REPORT.md и CSV показывает отдельно:

- A — исходный best 200/20;
- B — старый best или равное смешивание шага 0, выбранные только по Q4;
- C — новый 1000/100 с initial.

A→B измеряет допуск initial, B→C — новый бюджет и patience. Контроли:
matched_baseline из cache, equal_mixture, best_global отдельно по seed,
best_per_ticker, mean10/[1,5], mean80/[1,2,5,10,20] и constant каждого
режима сравнения. Fixed-выборы сверяются со старым selection.json.
Baseline Test MSE пересчитывается из cache и сверяется с исходным summary.

Основные столбцы — Q4 MSE, Test MSE и Test MAE. Абсолютная разница равна
MSE_method−MSE_control, Δ%=100×(MSE_method/MSE_control−1). Отрицательное
значение означает снижение ошибки. Сначала усредняются окна внутри
акции/seed, затем акции и seed с равным весом. Практическая ничья заранее
задана --tie-atol 1e-6 и --tie-rtol 1e-4.

CSV содержат все десять акций, seeds, исходные и новые результаты,
поакционные победы/поражения/ничьи, разброс между seed, квартальные
метрики и исключённые межквартальные окна. Общий test сохраняет все окна.
Парные интервалы для новых adaptive-vs-constant и adaptive-vs-best_global
усредняют seed внутри ticker/date и совместно пересэмплируют временные
блоки по акциям. 5000 выборок, seed 7, длина 20, чувствительность 10/40.
Интервалы условны на текущих акциях и пуле моделей.

Сохраняются source_audit.json, параметры и хэши в checkpoint,
per_fit.csv, effects.csv, summary.csv, per_seed.csv, per_ticker.csv,
per_ticker_seed_period.csv, weights_by_date.csv, weight_statistics.csv,
paired_block_intervals.csv, selection.json и графики. История каждого fit
лежит в router/{mode}/seed{seed}/history.json. Для весов показаны entropy,
exp(entropy), максимальный вес и L1-изменение во времени.
REPORT.md отдельно обсуждает пользу adaptive сверх constant, сравнение
с fixed global и согласованность направления по акциям/seed/кварталам.
При достижении лимита сходимость не объявляется установленной.
test_previously_inspected=true.

## Проверки

Из корня worktree с существующим Python, без установки библиотек:

```powershell
$env:PYTHONPATH="$PWD;$PWD\forecasting_task;$PWD\tests"
& C:\project_fin\FinTexTS\.venv\Scripts\python.exe -m unittest -v test_router_budget test_mean_router
git diff --check
& 'C:\Program Files\Git\bin\bash.exe' -n run_context_price_mean_router_budget.sh
```

Тесты проверяют шаг 0, ties, early stopping, возвращаемый best, resume
из шага 0 и с ненулевым patience, точный общий градиент и траекторию
относительно кода df0fdd2. Синтетический cache-only проход проверяет девять
fit и отчёт, блокирует чтение test на train, запрещает вызовы expert
training/cache generation и сверяет неизменность всех файлов source.

## Публикация в терминале VS Code Windows

```powershell
Set-Location C:\project_fin\FinTexTS\worktrees\context-price-mean-router-budget
git diff --check
git add forecasting_task/mean_router.py router_budget.py router_budget_source.py router_budget_report.py run_context_price_mean_router_budget.sh tests/test_router_budget.py ROUTER_BUDGET.md README.md
git diff --cached --stat
git commit -m "Add cache-only router budget experiment with initial checkpoint"
git push -u origin experiment/context-price-mean-router-budget
```

## Astra — одна команда подготовки и фонового запуска

Команда использует существующий контейнер и mount /workspace/variants.
При существующем worktree проверяются чистота и точный SHA. Несовпадение
останавливает команду с сохранением файлов. Для возобновления нового fit
добавьте --resume к аргументам run_context_price_mean_router_budget.sh.

```bash
(
  set -euo pipefail
  REPO="$HOME/projects/FinTexTS"
  WT="$HOME/projects/FinTexTS_compare/context-price-mean-router-budget"
  git -C "$REPO" fetch origin
  EXPECTED=$(git -C "$REPO" rev-parse origin/experiment/context-price-mean-router-budget)
  if [[ -e "$WT" ]]; then
    git -C "$WT" rev-parse --is-inside-work-tree >/dev/null
    [[ -z "$(git -C "$WT" status --porcelain)" ]] || { echo "Worktree has local changes; preserved"; exit 1; }
    [[ "$(git -C "$WT" rev-parse HEAD)" == "$EXPECTED" ]] || { echo "Existing worktree SHA differs; preserved"; exit 1; }
  else
    git -C "$REPO" worktree add --detach "$WT" "$EXPECTED"
  fi
  CTR=kondyurin-fintexts-compare
  CODE=/workspace/variants/context-price-mean-router-budget
  SOURCE=/auto/datasets/kondyurin/FinTexTS/runs/context_price_mean_router_q1_v1/FiLM
  OUTPUT=/auto/datasets/kondyurin/FinTexTS/runs/context_price_mean_router_budget_v1/FiLM
  docker exec "$CTR" test -f "$CODE/router_budget.py"
  if ! PROCESSES=$(docker top "$CTR" -eo pid,args); then
    echo "Cannot inspect container processes; no queue started"; exit 1
  fi
  if grep -E 'router_budget[.]py|mean_router_experiment[.]py|forecasting_task[.]run|run_.*(queue[.]py|[.]sh)' <<< "$PROCESSES"; then
    echo "An experiment queue is already running; no queue started"; exit 1
  else
    status=$?
    [[ "$status" == 1 ]] || { echo "Process filter failed; no queue started"; exit "$status"; }
  fi
  LOG="$OUTPUT/logs/queue_$(date -u +%Y%m%dT%H%M%S%N)_$$.log"
  docker exec "$CTR" mkdir -p "$OUTPUT/logs"
  docker exec -d -e CODE_COMMIT="$EXPECTED" -e PYTHONPATH="$CODE:$CODE/forecasting_task" \
    -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -w "$CODE" "$CTR" bash -c '
      set -euo pipefail
      set -o noclobber
      exec bash run_context_price_mean_router_budget.sh --source-root "$1" --output "$2" \
        --max-epochs 1000 --patience 100 >"$3" 2>&1
    ' _ "$SOURCE" "$OUTPUT" "$LOG"
  printf 'Started background CPU queue. Log: %s\n' "$LOG"
)
```

docker exec -d переживает разрыв SSH. Ctrl+C просмотра лога прекращает просмотр.
Проверить прогресс по последнему уникальному логу:

```bash
docker exec kondyurin-fintexts-compare bash -c '
  root=/auto/datasets/kondyurin/FinTexTS/runs/context_price_mean_router_budget_v1/FiLM
  latest=$(ls -1t "$root"/logs/queue_*.log | head -n 1)
  test -n "$latest" && exec tail -n 60 -F "$latest"
'
```

Отдельные plan и audit перед запуском:

```bash
docker exec -e PYTHONPATH=/workspace/variants/context-price-mean-router-budget:/workspace/variants/context-price-mean-router-budget/forecasting_task \
  -w /workspace/variants/context-price-mean-router-budget kondyurin-fintexts-compare python router_budget.py plan
docker exec -e CODE_COMMIT="$(git -C ~/projects/FinTexTS_compare/context-price-mean-router-budget rev-parse HEAD)" \
  -e PYTHONPATH=/workspace/variants/context-price-mean-router-budget:/workspace/variants/context-price-mean-router-budget/forecasting_task \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -w /workspace/variants/context-price-mean-router-budget \
  kondyurin-fintexts-compare python -u router_budget.py audit
```

Очередь сама формирует final. Повторный analyze сохраняйте под новым именем:

```bash
docker exec -e CODE_COMMIT="$(git -C ~/projects/FinTexTS_compare/context-price-mean-router-budget rev-parse HEAD)" \
  -e PYTHONPATH=/workspace/variants/context-price-mean-router-budget:/workspace/variants/context-price-mean-router-budget/forecasting_task \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -w /workspace/variants/context-price-mean-router-budget \
  kondyurin-fintexts-compare python -u router_budget.py analyze --max-epochs 1000 --patience 100 --report-name review2
```
