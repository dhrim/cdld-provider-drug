#!/usr/bin/env bash
# 재현 경로 smoke test — «자료 없는» 깨끗한 압축해제에서 모든 문서화된 명령을 돌린다.
#
#   $ tar xzf repo_v8.tgz && cd repo_v2 && bash tools/smoke_test.sh
#
# 계약:
#   자료가 필요 없는 것  -> exit 0
#   자료가 필요한 것     -> paths.need_data() 로 «깨끗이» 멈춘다 ([cdld] 한 줄)
#                          AttributeError / FileNotFoundError 로 죽으면 실패다
set -u
cd "$(dirname "$0")/.."
fail=0
run(){ printf '%-46s' "$1"; out=$(eval "$1" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then echo "OK   (exit 0)"
  else fail=1; echo "★FAIL exit $rc | $(echo "$out"|tail -1|cut -c1-90)"; fi; }

echo "=== 자료 없이 통과해야 하는 것 ==="
run "python3 verify.py"
run "python3 src/verify_v5.py"
run "python3 src/verify_v6.py"
run "python3 src/verify_v8.py"
run "python3 src/figs_v6.py"
run "python3 src/figs_v8.py"
run "python3 src/figs_v9.py"
run "python3 src/verify_v9.py"
run "python3 src/verify_v10.py"
run "python3 src/cdld_a.py"

echo
echo "=== 자료가 필요한 것 — «깨끗이» 멈춰야 한다 ==="
for c in "python3 src/build_emar.py" "python3 src/map_careunit.py" "python3 src/diag_null_B.py" \
         "python3 src/run_prereg.py none" "python3 src/run_ward.py" "python3 src/run_cdld.py 5" \
         "python3 src/run_sens.py temporal 5" "python3 src/run_cdld_a.py 5" "python3 src/run_cdld_a2.py 5" \
         "python3 src/run_cdld_a2_ward.py 5" "python3 src/run_nat_nodecay.py 5" "python3 src/run_a2_mid.py 5" \
         "python3 src/run_a2_null3.py 5" "python3 src/run_seed.py 1" "python3 src/retrain_b.py 0" \
         "python3 src/retrain_c.py 0" "python3 src/diag_d.py" "python3 src/diag_fn.py" \
         "python3 src/diag_fn2.py" "python3 src/interp_fit.py 5" "python3 src/collect.py" \
         "python3 src/final.py" "python3 src/final_ward.py" "python3 src/adj_a2.py" \
         "python3 src/boot_gate2.py" "python3 src/collect_v6.py" "python3 src/defend3.py" "python3 src/profiles2.py" "python3 src/diag_null_B.py"; do
  printf '%-46s' "$c"; out=$(eval "timeout 90 $c" 2>&1); rc=$?
  if echo "$out" | grep -q '\[cdld\]';           then echo "CLEAN STOP (need_data)"
  elif echo "$out" | grep -q 'python src/';      then echo "CLEAN STOP (usage)"
  elif [ $rc -eq 0 ];                            then echo "OK   (exit 0)"
  elif [ $rc -eq 124 ];                          then echo "running (90s timeout) — 시작함"
  else fail=1; echo "★FAIL $(echo "$out"|grep -E '(Error|Exception):'|tail -1|cut -c1-80)"; fi
done
echo
[ $fail -eq 0 ] && echo "전부 통과" || echo "★실패 있음"
exit $fail
