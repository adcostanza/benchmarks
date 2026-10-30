#!/bin/bash
# Calibration for the real-answer sets (results-real/), host side, CPU only. Per config:
#   typed-decision-bench          no fit split: cross-fitted on test, 5 folds stratified by task
#   procedural-typed-decisions    fit on the validation sample, scored on the test sample
#   calibrated-typed-decisions    fit on validation, scored on test and on heldout (PubMedQA)
# Writes results-real/<config>/calibration-<set>-<split>.json and the calibrated answers in
# results-real/<config>/calibrated/<set>-<split>/{T,T+B}.jsonl (read by real_answers.py).
#
# Usage: DATASETS_DIR=/path/to/datasets ./calibrate-real.sh [config ...]   default: both NVFP4 configs
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"  # relative paths: they are recorded in calibration-*.json
: "${DATASETS_DIR:?set DATASETS_DIR}"
cal() {
    uv run -q --no-project --with pyarrow --with numpy --with scipy ../scripts/decisions/calibrate.py \
        --data-dir "$DATASETS_DIR" "$@"
}
configs=("$@")
(( ${#configs[@]} )) || configs=(stock-nvfp4 swift-nvfp4)
for c in "${configs[@]}"; do
    r="results-real/$c"
    echo "== $c typed-decision-bench test (5-fold cross-fit)"
    cal --set typed-decision-bench --folds 5 --test-answers "$r/typed-decision-bench-test/pass1.jsonl" \
        --out "$r/calibration-typed-decision-bench-test.json" --rows-out "$r/calibrated/typed-decision-bench-test"
    echo "== $c procedural-typed-decisions test (fit on validation)"
    cal --set procedural-typed-decisions --fit-answers "$r/procedural-typed-decisions-validation/pass1.jsonl" --fit-split validation \
        --test-answers "$r/procedural-typed-decisions-test/pass1.jsonl" \
        --out "$r/calibration-procedural-typed-decisions-test.json" --rows-out "$r/calibrated/procedural-typed-decisions-test"
    for split in test heldout; do
        echo "== $c calibrated-typed-decisions $split (fit on validation)"
        cal --set calibrated-typed-decisions --fit-answers "$r/calibrated-typed-decisions-validation/pass1.jsonl" --fit-split validation \
            --test-answers "$r/calibrated-typed-decisions-$split/pass1.jsonl" --test-split "$split" \
            --out "$r/calibration-calibrated-typed-decisions-$split.json" --rows-out "$r/calibrated/calibrated-typed-decisions-$split"
    done
done
