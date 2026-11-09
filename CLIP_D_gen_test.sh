#!/usr/bin/env bash
# Cross-dataset generalization test: for every training preset, test on every preset
# (aligned case first). Each run's output is saved to logs/<detector>/<trn>__<tst>.log
#
# Usage: ./run_generalization.sh            # runs everything
#        DETECTORS="R50_TF" ./run_generalization.sh
#        FORCE=1 ./run_generalization.sh    # re-run even if the log already exists

set -u

DETECTORS=${DETECTORS:-"CLIP-D"}
LOG_ROOT=${LOG_ROOT:-logs}
FORCE=${FORCE:-0}

DATASETS=(
  seasons/autumn_TM01 seasons/spring_SP01 seasons/summer_SM01 seasons/winter_WN01
  style/bw_BW01 style/cinematic_CN01 style/cinematic2_CN11 style/filminspired_warmgold
  style/futuristic_FT01 style/vintage_VN01
  adaptive/blurbg_strong adaptive/blurbg_subtle adaptive/enhance_portrait
  adaptive/sky_bluedrama adaptive/subject_pop
  subject/landscape_LN01 subject/travel2_TR11 subject/travel_TR01
)

slug() { echo "${1//\//_}"; }

for det in $DETECTORS; do
#   mkdir -p "$LOG_ROOT/$det"
  for trn in "${DATASETS[@]}"; do
    # aligned case first, then all the others (training dataset fixed)
    ordered=("$trn")
    for d in "${DATASETS[@]}"; do [[ "$d" != "$trn" ]] && ordered+=("$d"); done

    for tst in "${ordered[@]}"; do
    #   log="$LOG_ROOT/$det/$(slug "$trn")__$(slug "$tst").log"
    #   if [[ -s "$log" && "$FORCE" != "1" ]]; then
    #     echo "[skip] $det | trn=$trn | test=$tst (log exists)"
    #     continue
    #   fi
      echo "[run ] $det | trn=$trn | test=$tst"
      python3 launcher.py \
        --detector "$det" \
        --phases test \
        --trn-dataset "$trn" \
        --dataset "$tst" \
        --weights-name lora_r4_qv --tf2k True --ft \
        || echo "[FAIL] $det | trn=$trn | test=$tst"
        # > "$log" 2>&1 \
        # || echo "[FAIL] $det | trn=$trn | test=$tst (see $log)"
    done
  done
done

echo "Done. Now run: python3 plot_generalization_matrix.py --log-root $LOG_ROOT"