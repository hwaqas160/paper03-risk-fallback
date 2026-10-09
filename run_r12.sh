cd /f/CLAUDE/AI3/paper03-risk-fallback
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
PY=F:/CLAUDE/AI1/shared/envs/unitraj/Scripts/python.exe
for w in sup collonly joint bonf; do
  $PY src/cswc2.py $w --out results/final/r12_$w.json > results/final/r12_$w.log 2> results/final/r12_$w.err
done
