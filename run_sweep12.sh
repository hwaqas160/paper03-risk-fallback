cd /f/CLAUDE/AI3/paper03-risk-fallback
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
F:/CLAUDE/AI1/shared/envs/unitraj/Scripts/python.exe src/cswc2.py sweep --out results/final/r12_sweep.json > results/final/r12_sweep.log 2> results/final/r12_sweep.err
