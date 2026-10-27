#!/bin/zsh
# Overnight launch of script 22 (Task 10 Step 2). Reruns script 22 until it exits 0, at most 25 attempts; each
# attempt skips the paths already written. An attempt that prints no "path " line for 600 s is killed (the libomp
# hang never exits on its own). Logs go to results/synthetic/rows/logs/ (gitignored).
# Launch from the repo root:  caffeinate -i nohup zsh scripts/22_relaunch.sh > /dev/null 2>&1 &
# Extra script 22 arguments pass through (e.g. --extra 200).
R=${0:A:h:h}
L=$R/results/synthetic/rows/logs
mkdir -p $L
cd $R
export OMP_NUM_THREADS=1 KMP_GTID_MODE=2
for a in $(seq 1 25); do
  log=$L/run_a$a.log
  echo "$(date '+%F %T') attempt $a started" >> $L/attempts.txt
  julia --gcthreads=1 --project=code -t 30 scripts/22_synthetic_run.jl "$@" > $log 2>&1 &
  jp=$!
  last=$(date +%s); n=0
  while kill -0 $jp 2>/dev/null; do
    sleep 10
    m=$(grep -c "^path " $log)
    if [ $m -ne $n ]; then n=$m; last=$(date +%s); fi
    if [ $(( $(date +%s) - last )) -gt 600 ]; then
      echo "$(date '+%F %T') attempt $a STALLED after $n paths (no path line for 600 s); killed" >> $L/attempts.txt
      kill -9 $jp
      break
    fi
  done
  wait $jp; rc=$?
  why=$(grep -m1 -E "signal \(?[0-9]+|OMP: Error|ERROR" $log)
  echo "$(date '+%F %T') attempt $a exit $rc after $(grep -c '^path ' $log) paths ${why}" >> $L/attempts.txt
  if [ $rc -eq 0 ]; then break; fi
done
echo "$(date '+%F %T') DONE: $(find $R/results/synthetic/rows -maxdepth 1 -name 'path_*.csv' | wc -l | tr -d ' ') path files" >> $L/attempts.txt
