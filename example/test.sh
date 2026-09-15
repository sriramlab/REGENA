#!/bin/bash
# REGENA toy examples (5,000 individuals x 10,000 SNPs).
# Run from this directory after building:  cd example && bash test.sh
set -euo pipefail

REGENA=${REGENA:-../build/REGENA}
nthreads=${NTHREADS:-4}

gen=test
covar=test.cov
env=test.env               # binary (0/1) environment

# 1) Box-Cox scan. scan_demo.pheno is additive on the log scale, but the environment
#    multiplies the trait, so the observed scale shows amplification GxE
#    (made by ../scripts/simulate_scale_gxe.py --h2 0.5 --a 0.5 --shift 0.5 --seed 1).
#    13 scales, lambda = -1, -0.75, ..., 2; one WIDE_TABLE row per lambda.
${REGENA} -g $gen -p scan_demo.pheno -c $covar -e $env -a single.annot \
    -m G+GxE+NxE -k 10 -jn 10 -t $nthreads -s 1 \
    -bx -bx-min -1 -bx-max 2 -bx-n 13 \
    -o scan_demo.out

# 2) A single scale. lambda = 1 is the observed scale (shifted and standardised).
${REGENA} -g $gen -p test.positive.pheno -c $covar -e $env -a single.annot \
    -m G+GxE+NxE -k 10 -jn 10 -t $nthreads -s 1 \
    -bx -bx-min 1 -bx-max 1 -bx-n 1 \
    -o test.lambda1.out

# 3) Partitioned model: 2 annotations, GxE partitioned by annotation (-eXa), 3 scales.
${REGENA} -g $gen -p test.positive.pheno -c $covar -e $env -a multi.annot -eXa \
    -m G+GxE+NxE -k 10 -jn 10 -t $nthreads -s 1 \
    -bx -bx-min 0 -bx-max 1 -bx-n 3 \
    -o test.partitioned.out

# 4) The scan in (1), driven by a config file (full key names instead of flags).
${REGENA} --config config.txt

# Per-lambda summary of (1) as a TSV (standard library only).
python3 ../scripts/extract_wide_table.py scan_demo.out > scan_demo.tsv

# Scale selection for (1): valid region, GxE tests, reported scale (needs numpy + pandas).
if python3 -c "import numpy, pandas" 2>/dev/null; then
    python3 ../scripts/regena_select_scale.py \
        --out scan_demo.out --pheno scan_demo.pheno --env $env \
        --trajectory scan_demo.trajectory.tsv | tee scan_demo.scale.tsv
else
    echo "numpy/pandas not found: skipping regena_select_scale.py"
fi
