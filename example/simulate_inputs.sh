#!/bin/bash
# Regenerate every input of the toy example from fixed seeds (needs python3 + numpy).
# The files shipped in this directory were written by exactly these commands; nothing
# here is derived from real genotype or phenotype data.
# Run from this directory:  cd example && bash simulate_inputs.sh
set -euo pipefail

# Genotypes (test.bed/.bim/.fam), covariates, environment and the two annotation files.
python3 ../scripts/simulate_example_data.py test --n 5000 --m 10000 --seed 1

# Trait that is additive on the log scale and multiplied by the environment.
python3 ../scripts/simulate_scale_gxe.py test test.env scan_demo.pheno \
    --h2 0.5 --a 0.5 --shift 0.5 --seed 1

# Same trait model without the environment effect: no GxE on any scale.
python3 ../scripts/simulate_scale_gxe.py test test.env test.positive.pheno \
    --h2 0.5 --a 0.5 --shift 0 --seed 2
