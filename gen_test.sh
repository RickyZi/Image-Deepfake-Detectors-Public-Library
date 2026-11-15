#!/usr/bin/env bash
# Cross-dataset generalization test: for every training preset, test on every preset
# (aligned case first).
#
# RESUMABLE: every (detector, trn, tst) combination that finishes successfully is
# recorded as an empty marker file in $STATE_ROOT/<detector>/<trn>__<tst>.done.
# If the VM shuts down abruptly, just re-launch the script: finished combinations
# are skipped and the run that was interrupted is executed again from scratch
# (a marker is only written AFTER launcher.py exits with status 0).
# Failed runs are not marked, so they are retried on the next launch.
#
# When a detector finishes with no failures, its marker folder is deleted to save
# space and a tiny $STATE_ROOT/<detector>.complete file is left, so a relaunch does
# not redo it.
#
# Usage: bash gen_test.sh                 # runs everything / resumes
#        DETECTORS="R50_nodown" bash gen_test.sh
#        FORCE=1 ./run_generalization.sh         # ignore markers, re-run everything
#        RESET=1 ./run_generalization.sh         # delete markers for the selected detectors, then run
#        STATE_ROOT=/data/gen_state ./run_generalization.sh   # keep state on a persistent disk
#
# Tip: run it under nohup/tmux so it also survives SSH disconnections:
#        nohup ./run_generalization.sh >> generalization.out 2>&1 &
#
# To auto-resume after a reboot, add a cron entry:
#        @reboot cd /path/to/repo && nohup ./run_generalization.sh >> generalization.out 2>&1 &

# run specific detectors with
# DETECTORS="R50_nodown" bash gen_test.sh
# DETECTORS="CLIP-D" bash gen_test.sh

set -u

# DETECTORS=${DETECTORS:-"R50_TF R50_nodown"}
DETECTORS=${DETECTORS:-"R50_TF R50_nodown CLIP-D"}
LOG_ROOT=${LOG_ROOT:-logs}
STATE_ROOT=${STATE_ROOT:-.gen_state}
FORCE=${FORCE:-0}
RESET=${RESET:-0}

DATASETS=(
  # adaptive
  adaptive/blurbg_strong
  adaptive/blurbg_subtle
  adaptive/enhance_portrait
  adaptive/sky_bluedrama
  adaptive/subject_pop

  # auto+
  auto+/retro-AR01
  auto+/retro-AR09

  # portraits
  portraits/bw-PB01
  portraits/deepskin-PD01-red
  portraits/deepskin-PD01-orange
  portraits/deepskin-PD01-yellow
  portraits/edgy-PE01
  portraits/group-PG01
  portraits/group-PG06
  portraits/group-PG07
  portraits/lightskin-PL01
  portraits/light-skin-PL10
  portraits/mediumskin-PM01
  portraits/medium-skin-PM11

  # seasons
  seasons/autumn_TM01
  seasons/spring_SP01
  seasons/summer_SM01
  seasons/winter_WN01

  # style
  style/bw_BW01
  style/cinematic_CN01
  style/cinematic2_CN11
  style/film_inspired_warmgold
  style/film_inspired_boldbw
  style/film_inspired_coolbw
  style/futuristic_FT01
  style/vintage_VN01

  # subject
  subject/landscape_LN01
  subject/travel_TR01
  subject/travel2_TR11
  subject/concerts-CT01
  subject/concerts-CT09
  subject/concerts-CT10
  subject/food-FD01
  subject/food-FD04
  subject/lifestyle-LF01
  subject/lifestyle-LF08
  subject/urbanarchitecture-UA01
  subject/urban-architecture-UA08

  # video
  video/creative-VCR01
  video/creative-VCR08
  video/creative-VCR12
)

slug() { echo "${1//\//_}"; }

# ---------------------------------------------------------------- single instance
# Prevents two copies (e.g. cron @reboot + manual launch) from running at once.
# mkdir -p "$STATE_ROOT"
# exec 9>"$STATE_ROOT/.lock"
# if ! flock -n 9; then
#   echo "[abort] another instance of this script is already running"
#   exit 1
# fi
# ---------------------------------------------------------------- multiple instances
# One lock per detector: two copies can run at once as long as they handle
# different detectors, but two copies on the same detector are refused.
mkdir -p "$STATE_ROOT"
for det in $DETECTORS; do
  exec {lock_fd}>"$STATE_ROOT/.lock_$det"
  if ! flock -n "$lock_fd"; then
    echo "[abort] another instance is already running $det"
    exit 1
  fi
done

# ---------------------------------------------------------------- optional reset
if [[ "$RESET" == "1" ]]; then
  for det in $DETECTORS; do
    echo "[reset] clearing markers for $det"
    rm -rf "${STATE_ROOT:?}/$det"
    rm -f "$STATE_ROOT/$det.complete"
  done
fi

# ---------------------------------------------------------------- detector args
# Fills the global array ARGS with the launcher arguments for a given detector.
# $1=detector  $2=train dataset  $3=test dataset
detector_args() {
  case "$1" in
    CLIP-D)
      ARGS=(--phases test --trn-dataset "$2" --dataset "$3"
            --weights-name lora_r4_qv --tf2k True --ft)
      ;;
    *)  
      ARGS=(--phases test --trn-dataset "$2" --dataset "$3"
            --ft --r50unfreezeL4)
      ;;
  esac
}

# ---------------------------------------------------------------- main loop
total_per_det=$(( ${#DATASETS[@]} * ${#DATASETS[@]} ))
n_skipped=0
n_run=0
n_failed=0
n_det_skipped=0

for det in $DETECTORS; do
  if [[ -e "$STATE_ROOT/$det.complete" && "$FORCE" != "1" ]]; then
    echo "[info] $det: already fully completed, skipping"
    n_det_skipped=$((n_det_skipped + 1))
    continue
  fi
  det_failed=0
  mkdir -p "$STATE_ROOT/$det"
  # mkdir -p "$LOG_ROOT/$det"

  already=$(find "$STATE_ROOT/$det" -maxdepth 1 -name '*.done' 2>/dev/null | wc -l)
  echo "[info] $det: $already/$total_per_det combinations already completed"

  for trn in "${DATASETS[@]}"; do
    # aligned case first, then all the others (training dataset fixed)
    ordered=("$trn")
    for d in "${DATASETS[@]}"; do [[ "$d" != "$trn" ]] && ordered+=("$d"); done

    for tst in "${ordered[@]}"; do
      marker="$STATE_ROOT/$det/$(slug "$trn")__$(slug "$tst").done"

      if [[ -e "$marker" && "$FORCE" != "1" ]]; then
        n_skipped=$((n_skipped + 1))
        continue   # silent skip, otherwise resuming prints thousands of lines
      fi

      echo "[run ] $det | trn=$trn | test=$tst"
      detector_args "$det" "$trn" "$tst"
      if python3 launcher.py --detector "$det" "${ARGS[@]}"
      then
        # Written only on success, via temp file + rename so it is atomic;
        # sync flushes it to disk in case the VM dies right after.
        tmp="$marker.tmp"
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "$tmp" && mv -f "$tmp" "$marker"
        sync
        n_run=$((n_run + 1))
      else
        echo "[FAIL] $det | trn=$trn | test=$tst (will be retried on next launch)"
        n_failed=$((n_failed + 1))
        det_failed=$((det_failed + 1))
      fi
    done
  done

  if (( det_failed == 0 )); then
    rm -rf "${STATE_ROOT:?}/$det"
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATE_ROOT/$det.complete"
    sync
    echo "[done] $det: all $total_per_det combinations finished, markers deleted"
  else
    echo "[warn] $det: $det_failed run(s) failed, markers kept so a relaunch retries only those"
  fi
done

echo "Done. ran=$n_run skipped(already done)=$n_skipped failed=$n_failed detectors_skipped=$n_det_skipped"
echo "Now run: python3 plot_generalization_matrix.py --log-root $LOG_ROOT"

# echo "plot gen matric for R50_TF"
# python3 utils/plot_gen_matrix.py --detector-dirs R50_TF_ft_unfreezeL4
# echo "done"

# echo "plot gen matric for R50_nodown"
# python3 utils/plot_gen_matrix.py --detector-dirs R50_nodown_ft_unfreezeL4
# echo "done"