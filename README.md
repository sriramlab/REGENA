# REGENA
**R**obust **E**stimator of **G**ene-**EN**vironment **A**rchitectures

Whether a gene-by-environment (GxE) interaction is present can depend on the scale the
phenotype is measured on. If the environment multiplies a trait, every genetic effect is
larger in one environment on the observed scale, while on the log scale there is no
interaction at all. REGENA fits the GENIE G+GxE+NxE variance-component model
([Pazokitoroudi et al. 2024](https://doi.org/10.1016/j.ajhg.2024.05.015)) on a grid of
Box-Cox transformed phenotypes in a single pass over the genotypes. At each scale it reports:

- the environment-specific genetic variances σ²_g0 and σ²_g1 (**amplification** when they differ)
- the cross-environment genetic correlation ρ_g (**effect heterogeneity** when ρ_g < 1)
- the environment-specific residual variances (NxE)

A companion script uses the scan to separate **scale-dependent** GxE, which some Box-Cox
scale removes, from **scale-independent** GxE, which no scale removes. It also picks the scale to report.

## Prerequisites
The following are required on a Linux or macOS machine to build REGENA:
```
g++ (or clang++)
cmake
make
```
The scale-selection script needs Python 3 with `numpy` and `pandas`.

## How to install

```
git clone https://github.com/<org>/REGENA.git
cd REGENA
mkdir build
cd build/
cmake ..
make
```

This produces the executable `build/REGENA`.

## Quick start

```
./REGENA -g <genotype prefix> -p <phenotype> -c <covariates> -e <binary env> -a <annotation> \
    -m G+GxE+NxE -k 10 -jn 100 -t 8 \
    -bx -bx-min -3 -bx-max 3 -bx-n 100 \
    -o trait.env.out

python3 scripts/regena_select_scale.py --out trait.env.out --pheno <phenotype> --env <binary env>
```

Alternatively, run REGENA with a newline-separated config file. The config file uses full key names
(e.g. `genotype=`, `use-boxcox=1`) instead of flags; see `example/config.txt`.
```
./REGENA --config <config file>
```

## Parameters

```
  -h, --help                  Show this message and exit
  -g, --genotype              The path of PLINK BED genotype file (prefix)
  -p, --phenotype             The path of phenotype file
  -c, --covariate             The path of covariate file
  -a, --annot                 The path of input annotation for partitioned heritability
  -o, --output                Output file path
  -e, --environment           The path of the binary environment file
  -m, --model                 Specification of the model. Currently there are 3 options:
                                1. additive genetic (G) effects only (arg: G)
                                2. additive genetic (G) and gene-environment (GxE) effects (arg: G+GxE)
                                3. additive genetic (G), gene-environment (GxE) and heterogeneous
                                   noise (NxE) effects (arg: G+GxE+NxE)
  -k, --num-vec               The number of random vectors (10 is recommended)
  -jn, --num-jack             The number of jackknife blocks (100 is recommended)
  -t, --nthreads              The number of threads for multithreading
  -s, --seed                  Seed for the random vectors (default: random)
  -v, --verbose               Verbose mode; output extra information (normal equations, sample sizes, ...)
  -eXa, --eXannot             Partition the GxE component with respect to the annotation file
                              (by default a single GxE component is fitted)
  -np, --norm-proj-pheno      Standardize the phenotype after regressing out covariates (default: 1)
  -i, --cov-add-intercept     Append an intercept to the covariates (default: 1)

Box-Cox scan
  -bx, --use-boxcox           Fit the model on a grid of Box-Cox transformed phenotypes
  -bx-min, --boxcox-min       Smallest lambda in the grid (default: -2)
  -bx-max, --boxcox-max       Largest lambda in the grid (default: 2)
  -bx-n, --boxcox-n           Number of evenly spaced lambda values (default: 100)
```

At each λ the phenotype is transformed with `(y^λ − 1)/λ` (`log y` at λ = 0), then
standardised to mean 0 and variance 1 across all non-missing individuals. This puts every
scale on one variance scale, so λ = 1 is the observed scale. `-bx-n 1` with
`-bx-min λ -bx-max λ` fits a single scale.

## File formats
```
Genotype:    PLINK BED format (.bed/.bim/.fam). SNPs with MAF = 0 must be excluded.
Phenotype:   header "FID IID name". With -bx the values must be strictly positive.
Covariate:   header "FID IID name_of_cov_1 ... name_of_cov_n".
Environment: header "FID IID name", one column coded 0/1.
Annotation:  M rows (M = number of SNPs) and K columns (K = number of annotations), no header.
             If SNP i belongs to annotation j there is "1" in row i and column j, otherwise "0".
             (delimiter is " ")

1) The number and order of individuals must be the same in the phenotype, genotype,
   environment and covariate files.
2) The number and order of SNPs must be the same in the bim file and the annotation file.
3) Individuals with NA or -9 in the phenotype or environment are excluded.
4) Run one phenotype per job. With -bx the phenotype is expanded into one column per λ,
   and every column shares its missingness.
```

## Output

REGENA writes the variance components, heritabilities and jackknife SEs for every fitted
scale to `<output>`. The same results appear in a tab-separated block between a `WIDE_TABLE` line
and a `*****` line, one row per λ:

| column | meaning |
|---|---|
| `phenotype`, `lambda` | row index and its Box-Cox λ |
| `gamma_bin<b>` | genetic covariance between environments (annotation `b`) |
| `sigma2_g_env<e>_bin<b>` | genetic variance in environment `e` |
| `sigma2_e_env<e>` | residual variance in environment `e` |
| `rho_bin<b>_env0-1` | genetic correlation between environments, γ / √(σ²_g0 σ²_g1) |
| `h2_g_env<e>_bin<b>`, `h2_e_env<e>` | heritability and residual fraction within environment `e` |
| `pval.rho`, `pval.sigma.diff`, `pval.h2.diff` | Wald tests of ρ_g = 1, σ²_g0 = σ²_g1 and h²_g0 = h²_g1 (annotation 0) |

Every estimate has a matching `_se` column. **Environment labels:** a 0/1 environment E is
expanded to the indicator columns [E, 1 − E]. So `env0` is the stratum with **E = 1** and `env1` is
the stratum with **E = 0**.

`scripts/extract_wide_table.py <output>` prints this block as a TSV.

## Scale selection

`scripts/regena_select_scale.py` reads the scan and the phenotype REGENA was run on:

1. **Valid region.** Assume z_i(λ) = f_λ(y_i) ~ N(μ, σ²). The profile log-likelihood is
   ℓ_p(λ) = −(n/2) log σ̂²(λ) + (λ − 1) Σ log y_i. A λ is kept if
   (ℓ_p(λ) − max ℓ_p)/n ≥ −0.5 and it lies in the contiguous block of the grid around the
   maximum. This excludes transformations with implausible tail behaviour. Scales where
   either σ²_g is not positive are also dropped.
2. **GxE tests** at every kept λ: amplification (H0: σ²_g0 = σ²_g1) and effect
   heterogeneity (H0: ρ_g = 1). S is the set of λ where neither test rejects at `--alpha`.
3. **Scale-dependent** (S non-empty): a Box-Cox scale removes the GxE. The reporting
   scale is the member of S that maximises (1 − w) σ²_g0 + w σ²_g1, where w is the mean of the
   environment column.
4. **Scale-independent** (S empty): no scale removes the GxE. The optimal scale λ* is the
   member of {λ : p_amp > 0.05 or p_rho > 0.05, min z(σ²_g) ≥ 2} closest to λ = 1, and
   λ* = 1 if that set is empty. `remaining_gxe` names the component still present at λ*.
   If σ²_g in one environment is not distinguishable from zero at λ = 1, S can be empty
   for reasons unrelated to scale; `weak_sigma2_g_at_observed` flags such pairs.

`--alpha` should be corrected for the number of trait × environment pairs tested.
`--trajectory` writes every per-λ statistic and test to a TSV.

## Toy example
Sample files are provided in the `example` directory. `scan_demo.pheno` is a trait with
h² = 0.5 on the log scale that the environment multiplies
(`scripts/simulate_scale_gxe.py`). On the observed scale that becomes amplification GxE. Run:
```
cd example
bash test.sh
```
The scan shows σ²_g0 ≠ σ²_g1 at λ = 1 (p = 1e-4). The two variances agree near
λ = 0 (p = 0.80), ρ_g stays at 1 at every scale, and the pair is classified as scale-dependent.

## Simulator
To simulate phenotypes with GxE effects on real genotypes, see https://github.com/sriramlab/Simulator.

## Citation
REGENA builds on GENIE and RHE-mc:
```
1. Ali Pazokitoroudi, Zhengtong Liu, Andrew Dahl, Noah Zaitlen, Saharon Rosset, Sriram Sankararaman.
   AJHG (2024); doi: 10.1016/j.ajhg.2024.05.015

2. Ali Pazokitoroudi, Yue Wu, Kathryn S. Burch, Kangcheng Hou, Aaron Zhou, Bogdan Pasaniuc, Sriram Sankararaman.
   Nature Communications (2020); doi: 10.1038/s41467-020-17576-9
```

## Version
```
v1.0.0
```
