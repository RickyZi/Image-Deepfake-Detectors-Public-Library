#!/usr/bin/env bash
# Run the test phase for every mode, detector, social and dataset preset.
# Order: all detectors on the BASELINE first, then all detectors on the FT models.
# Each run's output is saved to logs/<mode>/<detector>/<social>/<dataset>.log
#
# Resume: a run that finishes successfully leaves a "<log>.done" marker and is skipped
# on the next invocation. Failed or interrupted runs have no marker, so they are re-run.
#
# Usage: ./run_generalization.sh                          # baseline then ft, resuming
#        MODES="baseline" ./run_generalization.sh         # baseline only
#        MODES="ft" ./run_generalization.sh               # ft only
#        DETECTORS="R50_TF" ./run_generalization.sh
#        DETECTORS="CLIP-D" SOCIALS="facebook" ./run_generalization.sh
#        FORCE=1 ./run_generalization.sh                  # ignore markers, re-run everything

set -u

MODES=${MODES:-"baseline ft"}
DETECTORS=${DETECTORS:-"CLIP-D R50_TF R50_nodown"}
SOCIALS=${SOCIALS:-"facebook telegram twitter"}
LOG_ROOT=${LOG_ROOT:-logs}
FORCE=${FORCE:-0}

DATASETS=(
  seasons/autumn_TM01 seasons/spring_SP01 seasons/summer_SM01 seasons/winter_WN01
  style/bw_BW01 style/cinematic_CN01 style/cinematic2_CN11 style/film_inspired_warmgold
  style/film_inspired_boldbw style/film_inspired_coolbw style/futuristic_FT01 style/vintage_VN01
  adaptive/blurbg_strong adaptive/blurbg_subtle adaptive/enhance_portrait
  adaptive/sky_bluedrama adaptive/subject_pop
  subject/landscape_LN01 subject/travel2_TR11 subject/travel_TR01
)

slug() { echo "${1//\//_}"; }

# Build the launcher command for (mode, detector, dataset, social) into global array CMD.
build_cmd() {
  local mode="$1" det="$2" ds="$3" social="$4"
  CMD=(python launcher.py --detector "$det" --phases test)

  case "$det" in
    CLIP-D)
      if [[ "$mode" == "ft" ]]; then
        CMD+=(--weights-name lora_r4_qv --tf2k True --ft)
      else
        CMD+=(--tf2k True)
      fi
      ;;
    R50_TF|R50_nodown)
      [[ "$mode" == "ft" ]] && CMD+=(--ft --r50unfreezeL4)
      ;;
    *)
      return 1
      ;;
  esac

  CMD+=(--dataset "$ds" --social "$social")
}

n_run=0; n_skip=0; n_fail=0
failed=()

for mode in $MODES; do
  if [[ "$mode" != "baseline" && "$mode" != "ft" ]]; then
    echo "[warn] unknown mode '$mode' (use baseline or ft), skipping"
    continue
  fi

  echo
  echo "################ MODE: $mode ################"

  for det in $DETECTORS; do
    for social in $SOCIALS; do
      mkdir -p "$LOG_ROOT/$mode/$det/$social"

      for ds in "${DATASETS[@]}"; do
        log="$LOG_ROOT/$mode/$det/$social/$(slug "$ds").log"
        done_marker="$log.done"

        if [[ -f "$done_marker" && "$FORCE" != "1" ]]; then
          echo "[skip] $mode | $det | $social | $ds (already done)"
          n_skip=$((n_skip + 1))
          continue
        fi

        if ! build_cmd "$mode" "$det" "$ds" "$social"; then
          echo "[warn] unknown detector '$det', skipping"
          continue
        fi

        echo "[run ] $mode | $det | $social | $ds"
        rm -f "$done_marker"
        if "${CMD[@]}" > "$log" 2>&1; then
          touch "$done_marker"
          n_run=$((n_run + 1))
        else
          echo "[FAIL] $mode | $det | $social | $ds (see $log)"
          n_fail=$((n_fail + 1))
          failed+=("$mode|$det|$social|$ds")
        fi
      done
    done
  done
done

echo
echo "Done: $n_run run, $n_skip skipped, $n_fail failed. Logs in $LOG_ROOT/"
if [[ "$n_fail" -gt 0 ]]; then
  echo "Failed runs (re-run the script to retry them):"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi