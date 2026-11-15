#!/usr/bin/env bash
# Run the test phase for every mode, detector, social and dataset preset.
# Order: all detectors on the BASELINE first, then all detectors on the FT models.
# Each run's output is saved to $LOG_ROOT/<mode>/<detector>/<social>/<dataset>.log
#
# RESUMABLE: every (mode, detector, social, dataset) run that finishes successfully is
# recorded as a marker file in $STATE_ROOT/<mode>/<detector>/<social>/<dataset>.done.
# If the VM shuts down abruptly, just re-launch the script: finished runs are skipped
# and the run that was interrupted is executed again from scratch (a marker is only
# written AFTER launcher.py exits with status 0). Failed runs are not marked, so they
# are retried on the next launch.
#
# When a (mode, detector, social) block finishes with no failures, its marker folder is
# deleted to save space and a tiny $STATE_ROOT/<mode>/<detector>/<social>.complete file
# is left, so a relaunch does not redo it.
#
# Usage: ./run_generalization.sh                          # baseline then ft / resumes
#        MODES="baseline" ./run_generalization.sh         # baseline only
#        MODES="ft" ./run_generalization.sh               # ft only
#        DETECTORS="R50_TF" ./run_generalization.sh
#        DETECTORS="CLIP-D" SOCIALS="facebook" ./run_generalization.sh
#        FORCE=1 ./run_generalization.sh                  # ignore markers, re-run everything
#        RESET=1 ./run_generalization.sh                  # delete markers for the selected
#                                                         # modes/detectors, then run
#        STATE_ROOT=/data/gen_state ./run_generalization.sh   # keep state on a persistent disk
#
# Tip: run it under nohup/tmux so it also survives SSH disconnections:
#        nohup ./run_generalization.sh >> generalization.out 2>&1 &
#
# To auto-resume after a reboot, add a cron entry:
#        @reboot cd /path/to/repo && nohup ./run_generalization.sh >> generalization.out 2>&1 &

set -u
 
MODES=${MODES:-"baseline ft"}
DETECTORS=${DETECTORS:-"CLIP-D R50_TF R50_nodown"}
SOCIALS=${SOCIALS:-"facebook telegram twitter"}
LOG_ROOT=${LOG_ROOT:-logs}
STATE_ROOT=${STATE_ROOT:-.social_state}
FORCE=${FORCE:-0}
RESET=${RESET:-0}

DATASETS=(
  # # adaptive
  # adaptive/blurbg_strong
  # adaptive/blurbg_subtle
  # adaptive/enhance_portrait
  # adaptive/sky_bluedrama
  # adaptive/subject_pop

  # auto+
  auto+/retro_AR01
  auto+/retro_AR09

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
  portraits/lightskin-PL10
  portraits/mediumskin-PM01
  portraits/mediumskin-PM11

  # # seasons
  # seasons/autumn_TM01
  # seasons/spring_SP01
  # seasons/summer_SM01
  # seasons/winter_WN01

  # # style
  # style/bw_BW01
  # style/cinematic_CN01
  # style/cinematic2_CN11
  # style/film_inspired_warmgold
  # style/film_inspired_boldbw
  # style/film_inspired_coolbw
  # style/futuristic_FT01
  # style/vintage_VN01

  # # subject
  # subject/landscape_LN01
  # subject/travel_TR01
  # subject/travel2_TR11
  subject/concerts-CT01
  subject/concerts-CT09
  subject/concerts-CT10
  subject/food-FD01
  subject/food-FD04
  subject/lifestyle-LF01
  subject/lifestyle-LF08
  subject/urbanarchitecture-UA01
  subject/urbanarchitecture-UA08

  # video
  video/creative-VCR01
  video/creative-CR08
  video/creative-CR12
)

slug() { echo "${1//\//_}"; }
 
# ---------------------------------------------------------------- sanity check on modes
for mode in $MODES; do
  if [[ "$mode" != "baseline" && "$mode" != "ft" ]]; then
    echo "[abort] unknown mode '$mode' (use baseline or ft)"
    exit 1
  fi
done
 
# ---------------------------------------------------------------- multiple instances
# One lock per (mode, detector): two copies can run at once as long as they handle
# different modes/detectors, but two copies on the same one are refused.
mkdir -p "$STATE_ROOT"
for mode in $MODES; do
  for det in $DETECTORS; do
    exec {lock_fd}>"$STATE_ROOT/.lock_${mode}_${det}"
    if ! flock -n "$lock_fd"; then
      echo "[abort] another instance is already running $mode | $det"
      exit 1
    fi
  done
done
 
# ---------------------------------------------------------------- optional reset
if [[ "$RESET" == "1" ]]; then
  for mode in $MODES; do
    for det in $DETECTORS; do
      echo "[reset] clearing markers for $mode | $det"
      rm -rf "${STATE_ROOT:?}/$mode/$det"
    done
  done
fi
 
# ---------------------------------------------------------------- detector args
# Fills the global array ARGS with the launcher arguments for (mode, detector).
# $1=mode  $2=detector  $3=dataset  $4=social
detector_args() {
  local mode="$1" det="$2" ds="$3" social="$4"
  ARGS=(--detector "$det" --phases test)
 
  case "$det" in
    CLIP-D)
      if [[ "$mode" == "ft" ]]; then
        ARGS+=(--weights-name lora_r4_qv --tf2k True --ft)
      else
        ARGS+=(--tf2k True)
      fi
      ;;
    R50_TF|R50_nodown)
      [[ "$mode" == "ft" ]] && ARGS+=(--ft --r50unfreezeL4)
      ;;
    *)
      return 1
      ;;
  esac
 
  ARGS+=(--dataset "$ds" --social "$social")
}
 
# ---------------------------------------------------------------- main loop
total_per_block=${#DATASETS[@]}
n_skipped=0
n_run=0
n_failed=0
n_blocks_skipped=0
failed=()
 
for mode in $MODES; do
  echo
  echo "################ MODE: $mode ################"
 
  for det in $DETECTORS; do
    for social in $SOCIALS; do
      block="$mode | $det | $social"
      state_dir="$STATE_ROOT/$mode/$det/$social"
      complete_file="$STATE_ROOT/$mode/$det/$social.complete"
 
      if [[ -e "$complete_file" && "$FORCE" != "1" ]]; then
        echo "[info] $block: already fully completed, skipping"
        n_blocks_skipped=$((n_blocks_skipped + 1))
        continue
      fi
 
      block_failed=0
      mkdir -p "$state_dir" "$LOG_ROOT/$mode/$det/$social"
 
      already=$(find "$state_dir" -maxdepth 1 -name '*.done' 2>/dev/null | wc -l)
      echo "[info] $block: $already/$total_per_block already completed"
 
      for ds in "${DATASETS[@]}"; do
        marker="$state_dir/$(slug "$ds").done"
        log="$LOG_ROOT/$mode/$det/$social/$(slug "$ds").log"
 
        if [[ -e "$marker" && "$FORCE" != "1" ]]; then
          n_skipped=$((n_skipped + 1))
          continue   # silent skip, otherwise resuming prints thousands of lines
        fi
 
        if ! detector_args "$mode" "$det" "$ds" "$social"; then
          echo "[warn] unknown detector '$det', skipping"
          continue 3
        fi
 
        echo "[run ] $block | $ds"
        # Output goes to the terminal AND to the log; the exit status is launcher.py's
        # (not tee's), read from PIPESTATUS.
        python launcher.py "${ARGS[@]}" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
        if (( rc == 0 )); then
          # Written only on success, via temp file + rename so it is atomic;
          # sync flushes it to disk in case the VM dies right after.
          tmp="$marker.tmp"
          date -u +"%Y-%m-%dT%H:%M:%SZ" > "$tmp" && mv -f "$tmp" "$marker"
          sync
          n_run=$((n_run + 1))
        else
          echo "[FAIL] $block | $ds (exit $rc, see $log; will be retried on next launch)"
          n_failed=$((n_failed + 1))
          block_failed=$((block_failed + 1))
          failed+=("$block | $ds")
        fi
      done
 
      if (( block_failed == 0 )); then
        rm -rf "${state_dir:?}"
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "$complete_file"
        sync
        echo "[done] $block: all $total_per_block datasets finished, markers deleted"
      else
        echo "[warn] $block: $block_failed run(s) failed, markers kept so a relaunch retries only those"
      fi
    done
  done
done
 
echo
echo "Done. ran=$n_run skipped(already done)=$n_skipped failed=$n_failed blocks_skipped=$n_blocks_skipped"
echo "Logs in $LOG_ROOT/"
if (( n_failed > 0 )); then
  echo "Failed runs (re-run the script to retry them):"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi