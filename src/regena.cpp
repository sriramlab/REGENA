#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector> 
#include <random>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdlib> 
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/QR>
#include "time.h"

#include "auxillary.h"
#include "genotype.h"
#include "genomult.h"
#include "arguments.h"
#include "storage.h"
#include "matmult.h"
#include "io.h"
#include "std.h"
#include "functions.h"
#include "vectorfn.h"
#include "statsfn.h"

// #if SSE_SUPPORT == 1
// 	#define fastmultiply fastmultiply_sse
// 	#define fastmultiply_pre fastmultiply_pre_sse
// #else
// 	#define fastmultiply fastmultiply_normal
// 	#define fastmultiply_pre fastmultiply_pre_normal
// #endif

using namespace Eigen;
using namespace std;

// Storing in RowMajor Form
#ifdef USE_DOUBLE
int use_double = 1;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;
#else
int use_double = 0;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixXdr;
#endif

// If memeff = true, uses the memory efficient version
bool memeff = false;

// If opt1 = true, writes to a file after pass 1 
// Reads from file in pass 2
bool opt1 = true;

// If opt2 = true, reads groups of SNPs in block (termed read block)
// Number of SNPs in a read block can be controlled based on memory constraints by setting mem_Nsnp
bool opt2 = true ;
int mem_Nsnp = 10;
int mem_alloc = -1;

// Number of read blocks
// Approximately: Nsnp/mem_Nsnp
// Will  vary depending on the structure of jackknife blocks
int Nreadblocks; 

//ENV
// Partition the GxE component with respect to the annotations in the annotation file
bool Annot_x_E = false;

// Add environment variables to covariates
bool add_env_to_cov = true;

MatrixXdr Enviro;
MatrixXdr point_est_adj_gxe;
MatrixXdr jack_adj_gxe;


//Intermediate Variables
int blocksize;
int hsegsize;
double *partialsums;
double *sum_op;		
double *yint_e;
double *yint_m;
double **y_e;
double **y_m;


struct timespec t0;

clock_t total_begin = clock();
MatrixXdr pheno;
MatrixXdr mask;
MatrixXdr covariate;  
MatrixXdr Q;
MatrixXdr v1; //W^ty
MatrixXdr v2;            //QW^ty
MatrixXdr v3;    //WQW^ty
MatrixXdr new_pheno;

genotype g;
MatrixXdr geno_matrix; //(p,n)
int k,p,n;

MatrixXdr means; //(p,1)
MatrixXdr stds; //(p,1)
MatrixXdr sum2;
MatrixXdr sum;  

////////
//related to phenotype	
double y_sum; 
double y_mean;

options command_line_opts;

bool debug = false;
bool var_normalize = false;
bool memory_efficient = false;
bool missing = false;
bool fast_mode = true;
bool use_cov = false; 

// Jackknife-related variables
// Different approaches for defining the jackknife blocks
// jack_scheme  =
// 1: each block has constant number of SNPs (except the last). a block might span distinct chromosomes
// 2: each block has constant number of SNPs with each block contained within a single chromosome
// 3: each block has constant physical length with each block contained within a single chromosome. 
// For 3: physical length is specified in Mbs.
// Default : 1
int jack_scheme = 1;

// Number of jackknife blocks
// When jack_scheme = 2 or 3, this number will be set to 
// actual number based on the number of SNPs
int Njack = 1000;

// Size of jackknife block (in 1 Mb)
// Relevant for jack_scheme = 3
int jack_size;

int step_size;
int step_size_rem;

bool use_ysum;
bool keep_xsum;

///////
MatrixXdr jack;
MatrixXdr point_est;
MatrixXdr enrich_jack;
MatrixXdr enrich_point_est;


//define random vector z's
MatrixXdr  all_zb;
MatrixXdr  all_Uzb;
MatrixXdr res;
MatrixXdr XXz;
MatrixXdr Xy;
MatrixXdr yXXy;
MatrixXdr jack_yXXy;


///
//Matrix<int, Dynamic, Dynamic, RowMajor> gen;
MatrixXdr gen;
bool read_header;
//read variables
unsigned char mask2;
int wordsize;
unsigned int unitsperword;
int unitsize;
int nrow, ncol;
unsigned char *gtype;
int Nindv;
int Nsnp;
//
// Number of environmental variables
int Nenv = 0;

// Number of genetic bins (annotations)
// Set in read_annot
int Nbin = 1;

// Number of non-genetic bins
int nongen_Nbin = 0;

// Number of genetic bins
int gen_Nbin = 0;

// Total number of bins (related to the number of annotations)
// Number of VCs = T_Nbin  + 1 (sigma_e)
// (1 for G model with single annotation)
// For heterogeneous noise: T_Nbin = Nbin + nongen_Nbin + Nenv;
// For no heterogeneous noise: T_Nbin = Nbin + nongen_Nbin;
int T_Nbin ;

// Number of covariates
int Ncov;
int Nindv_mask;
int NC;

// Number of random vectors
int Nz = 10;


// Sum of number of SNPs assigned to each annotation 
// Can be greater than or less than or equal to the number of SNPs in the bim file (for overlapping annotations)
int Nsnp_annot = 0;

// Annotation matrix for binary annotations (Number of SNPs x Number of annotations) 
vector<vector<bool> > annot_bool;

// Number of SNPs in each bin
vector<int> len;

// Number of SNPs in each bin in a jackknife block
// Njack X (Total number of annotations (depends on the specific model used)
vector<vector<int> > jack_bin;
vector<int>  jack_block_size;
vector<int>  snp_to_jackblock;

vector<vector<int> > read_bin;

// Assigned in setup_read_blocks
vector<int> jack_to_read_block;
vector<int> snp_to_read_block;

vector<vector<int> > mem_bin;
vector<int>  mem_block_size;

vector <data> allgen;
vector <genotype> allgen_mail;
int global_snp_index;
bool use_mailman = true;

///reading single col annot
vector<int> SNP_annot;
bool use_1col_annot = false;


///Variables for reg out cov on both side of LM
bool both_side_cov = true;
MatrixXdr UXXz;
MatrixXdr XXUz;
MatrixXdr Xz;
MatrixXdr trVK;

//CHANGE (2/27)
bool gen_by_env;
//CHANGE(10/20)
bool hetero_noise;
bool cov_add_intercept;
int verbose;
bool trace;
bool use_dummy = false;
int nthreads;
MatMult mm;
std::ofstream outfile;
std::ofstream trace_file;
std::ofstream meta_file;

bool use_summary_genotypes = false;
string wgxsumfile;
string xsumfilepath;
string xsum_path;
string wgxsum_path;
string ysum_path;
std::ofstream xsum_ofs;
std::ofstream wgxsum_ofs;
std::ifstream wgxsum_ifs;
std::ifstream xsum_ifs;
std::ofstream ysum_ofs;
std::ifstream ysum_ifs;
string prefix;

int seed;
std::mt19937 seedr;
std::uniform_real_distribution<> udis (0,1);

int phenocount = 0;

std::vector<double> boxcox_lambdas;
bool use_boxcox = true;


std::istream& newline(std::istream& in)
{
	if ((in >> std::ws).peek() != std::char_traits<char>::to_int_type('\n')) {
		in.setstate(std::ios_base::failbit);
	}
	return in.ignore();
}

void initial_var(){
	p = g.Nsnp;
	n = g.Nindv;

	sum2.resize(p,1);
	sum.resize(p,1);

	if(!fast_mode && !memory_efficient){
		geno_matrix.resize(p,n);
		g.generate_eigen_geno(geno_matrix,var_normalize);
	}

	// Initial intermediate data structures
	blocksize = k;
	hsegsize = g.segment_size_hori;        // = log_3(n)
	int hsize = pow(3,hsegsize);
	int vsegsize = g.segment_size_ver;              // = log_3(p)
	int vsize = pow(3,vsegsize);

	partialsums = new double [blocksize];
	sum_op = new double[blocksize];
	yint_e = new double [hsize * blocksize];
	yint_m = new double [hsize * blocksize];
	memset (yint_m, 0, hsize * blocksize * sizeof(double));
	memset (yint_e, 0, hsize * blocksize * sizeof(double));

	y_e  = new double*[g.Nindv];
	for (int i = 0 ; i < g.Nindv ; i++) {
		y_e[i] = new double[blocksize];
		memset (y_e[i], 0, blocksize * sizeof(double));
	}

	y_m = new double*[hsegsize];
	for (int i = 0 ; i < hsegsize ; i++)
		y_m[i] = new double[blocksize];
}

// Upper-tail probability for standard normal (like scipy.stats.norm.sf)
inline double normal_sf(double z) {
    if (!std::isfinite(z)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    return 0.5 * std::erfc(z * inv_sqrt2);
}


// Forward declaration of wide-table printer
void print_wide_summary_table(
    const MatrixXdr &point_est_preserved,
    const MatrixXdr &point_se,
    const MatrixXdr &SEjack_adj_gxe,
    const std::vector<double> &rho_hat_flat,
    const std::vector<double> &rho_se_flat);


// Needed for the memory efficient vesion (opt1 = true && opt2 = true).
// Needed to handle the cases where jackknife blocks do not all have equal number of SNPs
// Sets up data structures that map SNPs and jackknife blocks to read blocks
void setup_read_blocks ()  {

	int snpindex = 0;
	int blockindex = 0 ;

    // How many read blocks in each jackknife block
	jack_to_read_block.resize (Njack);		

    // Which read block each SNP belongs to
	snp_to_read_block.resize (Nsnp);

	for (int i = 0; i < Njack; i++) { 
		int jack_Nsnp = jack_block_size[i];
		int read_max_Nsnp = jack_Nsnp > mem_Nsnp ? mem_Nsnp: jack_Nsnp;
		int num_blocks = jack_Nsnp/read_max_Nsnp;

		jack_to_read_block[i] = num_blocks;
		for (int j = 0 ; j < num_blocks * read_max_Nsnp; j++) {
			snp_to_read_block[snpindex] = blockindex + j/read_max_Nsnp;
			snpindex ++ ;
		}
		for (int j = num_blocks * read_max_Nsnp; j < jack_Nsnp; j++){
			snp_to_read_block[snpindex] = blockindex + num_blocks - 1;
			snpindex ++ ;
		}	
		blockindex += num_blocks;
	}
	Nreadblocks = blockindex; 

	if (verbose >= 2) { 
		cout << "Number of read blocks = " << Nreadblocks << endl;

		if (verbose >= 4){
			vectorfn::printvector (jack_to_read_block); cout << endl;
			vectorfn::printvector (snp_to_read_block); cout << endl;
		}
	}
}

void setup_boxcox_lambdas() {
    if (!command_line_opts.use_boxcox) return;

    double lo = command_line_opts.boxcox_lambda_min;
    double hi = command_line_opts.boxcox_lambda_max;
    int    n  = command_line_opts.boxcox_n_points;

    boxcox_lambdas.clear();

    if (n <= 0) {
        std::cerr << "ERROR: boxcox_n_points must be > 0\n";
        exit(1);
    }

    if (n == 1) {
        boxcox_lambdas.push_back(lo);
    } else {
        double step = (hi - lo) / (n - 1);
        for (int k = 0; k < n; ++k) {
            boxcox_lambdas.push_back(lo + k * step);
        }
    }
}


void set_metadata() {
	wordsize = sizeof(char) * 8;
	unitsize = 2;
	unitsperword = wordsize / unitsize;
	mask2 = 0;
	for (int i = 0 ; i < unitsize; i++)
		mask2 = mask2 |(0x1 << i);
	nrow = Nsnp;
	ncol = ceil(1.0 * Nindv / unitsperword);
}

static void overwrite_additive_with_gamma_bin(int full_idx) {
    if (!gen_by_env) return;
    if (Nenv < 2) {
        std::cerr << "gamma-bin: need two env columns [E, 1-E]. Nenv=" << Nenv << std::endl;
        std::exit(1);
    }

    // Determine which two GxE bins to subtract
    int gxe_idx[2];
    if (Annot_x_E) {
        gxe_idx[0] = gen_Nbin + 0 * gen_Nbin + full_idx;
        gxe_idx[1] = gen_Nbin + 1 * gen_Nbin + full_idx;
    } else {
        gxe_idx[0] = gen_Nbin + 0;
        gxe_idx[1] = gen_Nbin + 1;
    }

    // Safety check
    if (full_idx < 0 || full_idx >= gen_Nbin) {
        std::cerr << "[ERR] full_idx out of range: " << full_idx
                  << " (gen_Nbin=" << gen_Nbin << ")\n";
        std::exit(1);
    }
    for (int t = 0; t < 2; t++) {
        if (gxe_idx[t] < 0 || gxe_idx[t] >= T_Nbin) {
            std::cerr << "[ERR] bad GxE index: " << gxe_idx[t]
                      << " (T_Nbin=" << T_Nbin << ")\n";
            std::exit(1);
        }
    }

    // --- memeff layout ---
    if (memeff) {
        int stride = 2 * Nz;
        int cols = XXz.cols();

        for (int z = 0; z < Nz; z++) {
            int oj_full = full_idx * stride + z;
            int ow_full = full_idx * stride + (Nz + z);

            if (oj_full >= cols || ow_full >= cols) {
                std::cerr << "[ERR] full_idx column out of range\n";
                std::exit(1);
            }

            // subtract the two gxe contributions
            for (int t = 0; t < 2; t++) {
                int gi = gxe_idx[t];
                int oj_gi = gi * stride + z;
                int ow_gi = gi * stride + (Nz + z);

                if (oj_gi >= cols || ow_gi >= cols) {
                    std::cerr << "[ERR] gxe_idx column out of range\n";
                    std::exit(1);
                }

                XXz.col(oj_full) -= XXz.col(oj_gi);
                XXz.col(ow_full) -= XXz.col(ow_gi);

                if (both_side_cov) {
                    UXXz.col(oj_full) -= UXXz.col(oj_gi);
                    UXXz.col(ow_full) -= UXXz.col(ow_gi);
                    XXUz.col(oj_full) -= XXUz.col(oj_gi);
                    XXUz.col(ow_full) -= XXUz.col(ow_gi);
                }
            }

            if (!std::isfinite(XXz(0, oj_full)) || !std::isfinite(XXz(0, ow_full))) {
                std::cerr << "[ERR] non-finite values after subtraction at bin="
                          << full_idx << " z=" << z << "\n";
                std::exit(1);
            }
        }

        for (int c = 0; c < yXXy.cols(); c++) {
            double newval = yXXy(full_idx, c) - (yXXy(gxe_idx[0], c) + yXXy(gxe_idx[1], c));
            if (!std::isfinite(newval)) {
                std::cerr << "[ERR] non-finite yXXy after subtraction (memeff)\n";
                std::exit(1);
            }
            yXXy(full_idx, c) = newval;
        }
    }

    // --- non-memeff layout ---
    else {
        int stride = (Njack + 1) * Nz;
        int cols = XXz.cols();

        for (int j = 0; j <= Njack; j++) {
            for (int z = 0; z < Nz; z++) {
                int ofull = full_idx * stride + j * Nz + z;
                if (ofull >= cols) {
                    std::cerr << "[ERR] full_idx column out of range (non-memeff)\n";
                    std::exit(1);
                }

                for (int t = 0; t < 2; t++) {
                    int gi = gxe_idx[t];
                    int ogi = gi * stride + j * Nz + z;
                    if (ogi >= cols) {
                        std::cerr << "[ERR] gxe_idx column out of range (non-memeff)\n";
                        std::exit(1);
                    }

                    XXz.col(ofull) -= XXz.col(ogi);
                    if (both_side_cov) {
                        UXXz.col(ofull) -= UXXz.col(ogi);
                        XXUz.col(ofull) -= XXUz.col(ogi);
                    }
                }

                if (!std::isfinite(XXz(0, ofull))) {
                    std::cerr << "[ERR] non-finite values after subtraction (non-memeff) "
                              << "bin=" << full_idx << " j=" << j << " z=" << z << "\n";
                    std::exit(1);
                }
            }
		}

		if (phenocount <= 1) {
			// old scalar behaviour
			yXXy.row(full_idx) -= yXXy.row(gxe_idx[0]);
			yXXy.row(full_idx) -= yXXy.row(gxe_idx[1]);
			if (!yXXy.row(full_idx).array().isFinite().all()) {
				std::cerr << "[ERR] non-finite yXXy after subtraction (non-memeff)\n";
				std::exit(1);
			}
		} else {
			// multi-phenotype: rows are (bin * phenocount + k)
			for (int k = 0; k < phenocount; ++k) {
				int r_full = full_idx * phenocount + k;
				int r_g0   = gxe_idx[0] * phenocount + k;
				int r_g1   = gxe_idx[1] * phenocount + k;

				if (r_full >= yXXy.rows() || r_g0 >= yXXy.rows() || r_g1 >= yXXy.rows()) {
					std::cerr << "[ERR] yXXy row out of range in gamma rewrite\n";
					std::exit(1);
				}

				yXXy.row(r_full) -= yXXy.row(r_g0);
				yXXy.row(r_full) -= yXXy.row(r_g1);

				if (!yXXy.row(r_full).array().isFinite().all()) {
					std::cerr << "[ERR] non-finite yXXy after subtraction (non-memeff, multi-phen)\n";
					std::exit(1);
				}
			}
		}
    }

    if (verbose >= 1) {
        std::cout << "Replaced additive bin " << full_idx << " with gamma (cross) bin.\n";
    }
}


inline MatrixXdr build_Xl_mask(int gen_Nbin, int nongen_Nbins, int Nenv)
{
    const int G = gen_Nbin;        // # gammas (annotation bins)
    const int M = nongen_Nbins;    // # gxe bins (annotation x env, typically G * Nenv)
    const int E = Nenv;            // # nxe bins (one per env)
    const int P = G + M + E;       // total size

    MatrixXdr mask = MatrixXdr::Zero(P, P);

    // --- Gamma block: orthogonal to all others -> keep only diagonals
    for (int i = 0; i < G; ++i) mask(i, i) = 1.0;

    // --- gxe diagonals
    for (int i = 0; i < M; ++i) mask(G + i, G + i) = 1.0;

    // --- nxe diagonals
    for (int i = 0; i < E; ++i) mask(G + M + i, G + M + i) = 1.0;

    // If no environments or no gxe bins, nothing else to do
    if (E <= 0 || M <= 0) return mask;

    // Infer how many gxe bins per environment block.
    // Typical case: M == G * E -> bins_per_env == G.
    const int bins_per_env = (E > 0 ? (M / E) : 0);

    if (bins_per_env <= 0) {
        // Fallback: couple min(M, E) in order, as a safe default
        const int K = std::min(M, E);
        for (int i = 0; i < K; ++i) {
            const int gi = G + i;         // gxe_i
            const int ei = G + M + i;     // nxe_i
            mask(gi, ei) = 1.0;
            mask(ei, gi) = 1.0;
        }
        return mask;
    }

    // For each environment e:
    // 1) Set all cross terms among gxe bins within that env block to 1
    // 2) Couple all gxe bins in that block with the corresponding nxe_e
    for (int e = 0; e < E; ++e) {
        const int ei = G + M + e; // index of nxe_e

        // Compute the gxe block [gxe_start, gxe_end] for this environment.
        // Cap to avoid overflow if M is not an exact multiple of E.
        const int gxe_start = G + e * bins_per_env;
        const int gxe_end   = std::min(G + M, gxe_start + bins_per_env) - 1;
        if (gxe_start > gxe_end) continue;

        // (1) All cross-terms among gxe bins within the same env = 1
        for (int i = gxe_start; i <= gxe_end; ++i) {
            for (int j = gxe_start; j <= gxe_end; ++j) {
                mask(i, j) = 1.0;
            }
        }

        // (2) Couple every gxe in this env to its nxe_e
        for (int i = gxe_start; i <= gxe_end; ++i) {
            mask(i, ei) = 1.0;
            mask(ei, i) = 1.0;
        }
    }

    return mask;
}


void genotype_stream_pass_mem_efficient (string name){
	MatrixXdr output_yXXy;
	MatrixXdr output_XXz;
	MatrixXdr tmpoutput_XXz;		
	MatrixXdr output_XXUz;
	MatrixXdr output_env;
	MatrixXdr output;
	double temp_yXXy;


	if (hetero_noise == true) {
		T_Nbin = Nbin + nongen_Nbin + Nenv;

		output_XXz = MatrixXdr::Zero(Nindv,T_Nbin * Nz);
		tmpoutput_XXz = MatrixXdr::Zero(Nindv, Nz);

		if(both_side_cov == true){
			output_XXUz = MatrixXdr::Zero(Nindv, T_Nbin * Nz);
		}
		output_yXXy = MatrixXdr::Zero(T_Nbin, 1);
		jack_yXXy = MatrixXdr::Zero(T_Nbin, Njack);

	} else {
		T_Nbin = Nbin + nongen_Nbin;

		output_XXz = MatrixXdr::Zero(Nindv, T_Nbin * Nz);
		tmpoutput_XXz = MatrixXdr::Zero(Nindv, Nz);

		if(both_side_cov == true){
			output_XXUz = MatrixXdr::Zero(Nindv, T_Nbin * Nz);
		}
		output_yXXy = MatrixXdr::Zero(T_Nbin, 1);
		jack_yXXy = MatrixXdr::Zero(T_Nbin, Njack);

	}

	if (verbose >= 3) { 		
		cout << "both_side_cov = " << both_side_cov << endl;
		cout << "cov_add_intercept = " << cov_add_intercept << endl;
		cout << "T_Nbin = " << T_Nbin << " Nbin = " << Nbin << " nongen_Nbin = " << nongen_Nbin << endl;
		cout << "Nz = " << Nz << endl;
		cout << "tmpoutput(" << tmpoutput_XXz.rows() <<"," << tmpoutput_XXz.cols() << ") "<< tmpoutput_XXz.sum() << endl;
		cout << "output(" << output_XXz.rows() <<"," << output_XXz.cols() << ") "<< output_XXz.sum() << endl;
	}

	if (opt1){ 
		string prefix = command_line_opts.OUTPUT_FILE_PATH;

		if (!use_summary_genotypes) {
			xsum_path = prefix + ".xsum";
			xsum_ofs.open(xsum_path.c_str(), std::ios_base::out);
		}
		if (use_ysum) {
			ysum_path = prefix + ".ysum";
			ysum_ofs.open(ysum_path.c_str(), std::ios_base::out);
		}
	}

	ifstream ifs (name.c_str(), ios::in|ios::binary);
	if (!ifs.is_open()){
		cerr << "Error reading file "<< name <<endl;
		exit(1);
	}

	read_header = true;
	global_snp_index=-1;

	MatrixXdr vec1;
	MatrixXdr w1;
	MatrixXdr w2;
	MatrixXdr w3;

	MatrixXdr A_trs(T_Nbin,T_Nbin);
	MatrixXdr b_trk(T_Nbin,1);
	MatrixXdr c_yky(T_Nbin,1);

	MatrixXdr X_l(T_Nbin,T_Nbin);
	MatrixXdr Y_r(T_Nbin,1);
	MatrixXdr B1;
	MatrixXdr B2;
	MatrixXdr C1;
	MatrixXdr C2;
	double trkij;
	double yy = (pheno.array() * pheno.array()).sum();

	if(both_side_cov == true){
		yy = (new_pheno.array() * new_pheno.array()).sum();
	}

	Nindv_mask = mask.sum();
	if(both_side_cov == true)
		NC = Nindv_mask - Ncov;
	else
		NC = Nindv_mask;

	MatrixXdr herit;

	// Matrix of jacknife estimates
	// Rows: statistics (=variance components including sigma_e). Hence number of rows = T_Nbin + 1
	// Columns: jackknife estimates
	jack.resize(T_Nbin, Njack);

	// Matrix of variance components
	// Includes sigma_e. Hence number of entries = T_Nbin + 1
	point_est.resize(T_Nbin, 1);

	point_est_adj_gxe = MatrixXdr::Zero(T_Nbin, 1);
	jack_adj_gxe = MatrixXdr::Zero(T_Nbin, Njack);

	enrich_jack.resize(T_Nbin, Njack);
	enrich_point_est.resize(T_Nbin, 1);


	MatrixXdr h1;
	MatrixXdr h2;
	MatrixXdr h3;

	double trkij_res1;
	double trkij_res2;
	double trkij_res3;
	double tk_res;

	int global_block_index = 0;
	for (int jack_index = 0 ; jack_index < Njack ; jack_index++){

		if (hetero_noise == true) {
			output_XXz = MatrixXdr::Zero(Nindv,T_Nbin * Nz);

			if(both_side_cov == true){
				output_XXUz = MatrixXdr::Zero(Nindv, T_Nbin * Nz);
			}
			output_yXXy = MatrixXdr::Zero(T_Nbin,1);
		} else {
			T_Nbin = Nbin + nongen_Nbin;

			output_XXz = MatrixXdr::Zero(Nindv, T_Nbin * Nz);

			if(both_side_cov == true){
				output_XXUz = MatrixXdr::Zero(Nindv, T_Nbin * Nz);
			}
			output_yXXy = MatrixXdr::Zero(T_Nbin, 1);
		}	

		int jack_Nsnp = jack_block_size [jack_index];	
		int read_Nsnp = jack_Nsnp > mem_Nsnp ? mem_Nsnp: jack_Nsnp;
		int num_blocks = jack_to_read_block [jack_index];
		int rem = jack_Nsnp - num_blocks * read_Nsnp;

		if (verbose >= 1) 
			cout << "************Pass 1: Reading jackknife block " << jack_index << " ************" <<endl;

		cout << "Pass 1: Reading jackknife block " << jack_index << endl;
		cout << "Dividing jackknife block " << jack_index << " into " << num_blocks << " read blocks each with " << read_Nsnp << " SNPs ";
		if (rem > 0)
			cout << " (last block has " << read_Nsnp + rem << " SNPs)";
		cout << endl;

		for (int block_index = 0; block_index < num_blocks; block_index ++, global_block_index++ ){
			read_Nsnp += (block_index < (num_blocks - 1))? 0 : rem;

			cout << "Pass 1: Reading SNP block " << block_index << " in jackknife block " << jack_index << ", global SNP block index " << global_block_index << endl;

			if (verbose >= 1)
				cout << "read_Nsnp = " << read_Nsnp << endl;

			if(use_mailman == true){
				if (verbose >= 1) { 
					cout << "Number of SNPs in each bin in jackknife block "<< jack_index << endl;
					vectorfn::printvector (jack_bin[jack_index]); cout << endl;
				}
				for (int i = 0 ; i < Nbin ; i++){
					allgen_mail[i].segment_size_hori = floor(log(Nindv) / log(3)) - 2 ;
					allgen_mail[i].Nsegments_hori = ceil(read_bin[global_block_index][i]*1.0 / (allgen_mail[i].segment_size_hori * 1.0));
					allgen_mail[i].p.resize(allgen_mail[i].Nsegments_hori,vector<int>(Nindv));

					allgen_mail[i].index = 0;
					allgen_mail[i].Nsnp = read_bin[global_block_index][i];
					allgen_mail[i].Nindv = Nindv;

					allgen_mail[i].columnsum.resize(read_bin[global_block_index][i],1);
					for (int index_temp = 0 ; index_temp < read_bin[global_block_index][i];index_temp++)
						allgen_mail[i].columnsum[index_temp]=0;
				}
			} else {
				for (int k = 0 ; k < Nbin ; k++){
					allgen[k].gen.resize(read_bin[global_block_index][k],Nindv);
					// number of SNPs in this annotation (bin) in this jackknife block
					allgen[k].index = 0;
				}
			}

			if(use_1col_annot == true)
				read_bed_1colannot(ifs, missing, read_Nsnp);
			else
				read_bed2(ifs, missing, read_Nsnp);
			read_header = false;

			for (int bin_index = 0 ; bin_index < Nbin ; bin_index++){
				int num_snp;
				if (use_mailman == true)
					num_snp = allgen_mail[bin_index].index;
				else
					num_snp = allgen[bin_index].index;

				if (verbose >= 2)
					cout << "Number of SNPs in bin " << bin_index << "  = " << num_snp << endl; 

				// The case where the read block for this bin is empty
				// Can skip this bin unless this is the last read block inside jackknife block
				// In that case, we need to update the whole-genome statistics computed so far 
				// We also need to write the relevant statistics to the file
				if (num_snp == 0)  {
					if (block_index == num_blocks - 1){

						if (!use_summary_genotypes){
						for (int z_index = 0 ; z_index < Nz ; z_index++){
							XXz.col((bin_index * 2 * Nz) + Nz + z_index) += output_XXz.col(bin_index * Nz + z_index);   /// save whole sample contribution

							if(both_side_cov == true) {
								vec1 = output_XXz.col(bin_index * Nz + z_index);
								w1 = covariate.transpose() * vec1;
								w2 = Q * w1;
								w3 = covariate * w2;
								UXXz.col((bin_index * 2 * Nz) + Nz + z_index) += w3;
								UXXz.col((bin_index * 2 * Nz) + z_index) += w3;
							}
						}
						}

						yXXy(bin_index,1) += output_yXXy(bin_index, 0);


						if (opt1) {
							// Check the number of SNPs in this bin within the jackknife block
							int tmpsnp = jack_bin[jack_index][bin_index];
							// If the bin is also empty for the jackknife block 
							// Only need to write the number of SNPs (=0)
							// So that this bin within the jackknnife block can be skipped in pass number 2
							if (use_summary_genotypes){
								double temp_yXXy = output_yXXy(bin_index, 0);
								
								if (use_ysum)
									ysum_ofs.write((char *) (&temp_yXXy), sizeof(double));
								else
									jack_yXXy(bin_index, jack_index) = temp_yXXy;
							} else {
							xsum_ofs.write((char *) (&tmpsnp), sizeof(int));
							if (tmpsnp != 0){

								// Handle non-empty bin within jackknife block
								// First write XXz, XXUz.
								MatrixXdr tmpoutput = output_XXz.block (0, bin_index * Nz, output_XXz.rows(), Nz);	

								write_matrix (xsum_ofs, tmpoutput);
								if (both_side_cov == true ) {
									MatrixXdr tmpoutput = output_XXUz.block (0, bin_index * Nz, output_XXUz.rows(), Nz);	
									write_matrix (xsum_ofs, tmpoutput);
								}

								double temp_yXXy = output_yXXy(bin_index, 0);
								if (use_ysum)
									ysum_ofs.write((char *) (&temp_yXXy), sizeof(double));
								else
									jack_yXXy(bin_index, jack_index) = temp_yXXy;
							}
							}
						}
					}
					continue; 
				} // End of code to handle empty blocks

				stds.resize(num_snp,1);
				means.resize(num_snp,1);

				if(use_mailman == true){
					for (int i = 0 ; i < num_snp ; i++)
						means(i,0) = (double)allgen_mail[bin_index].columnsum[i]/Nindv;
				} else	  
					means = allgen[bin_index].gen.rowwise().mean();


				for (int i = 0 ; i < num_snp ; i++)
					stds(i,0) = 1 / sqrt((means(i,0) * (1-(0.5 * means(i,0)))));

				if (use_mailman == true){
					g = allgen_mail[bin_index];
					g.segment_size_hori = floor(log(Nindv) / log(3)) - 2 ;
					g.Nsegments_hori = ceil(read_bin[global_block_index][bin_index]*1.0 / (g.segment_size_hori * 1.0));
					g.p.resize(g.Nsegments_hori,vector<int>(Nindv));
					initial_var();
				} else {
					gen = allgen[bin_index].gen;
				}

				// Setup data structures for mailman
				mm = MatMult(g, gen, debug, var_normalize, memory_efficient, missing, use_mailman, nthreads, Nz);

				if (!use_summary_genotypes){
				tmpoutput_XXz = compute_XXz(num_snp, all_zb);

				for (int z_index = 0 ; z_index < Nz ; z_index++)
					output_XXz.col(bin_index * Nz + z_index) = output_XXz.col(bin_index * Nz + z_index) + tmpoutput_XXz.col(z_index);

				if (verbose >= 3) {
					cout << "all_zb = " << all_zb.rows() << "," << all_zb.cols() << "\t" << all_zb.sum () << endl;
					cout << "Pass 1: " << jack_index << " " << block_index << " " << bin_index << "\ttmpoutput(" << tmpoutput_XXz.rows() <<"," << tmpoutput_XXz.cols() << ") "<< tmpoutput_XXz.sum() << endl;
					cout << "Pass 1: " << jack_index << " " << block_index << " " << bin_index << "\toutput(" << output_XXz.rows() <<"," << output_XXz.cols() << ") "<< output_XXz.sum() << endl;
				}

				if (opt1 && block_index == (num_blocks - 1)) {
					int tmpsnp = jack_bin[jack_index][bin_index];
					MatrixXdr tmpoutput = output_XXz.block (0, bin_index * Nz, output_XXz.rows(), Nz);	
					if (verbose >= 3) {
						cout << "output(" << tmpoutput.rows() <<"," << tmpoutput.cols() << ") "<< tmpoutput.sum() << endl;
					}

					xsum_ofs.write((char *) (&tmpsnp), sizeof(int));
					write_matrix (xsum_ofs, tmpoutput);
				}

				// begin gxe computations
				// This code block has not been tested for the memory optimized setting
				MatrixXdr scaled_pheno;
				if (gen_by_env == true) {
				// This code block not tested
					for (int env_index = 0 ; env_index < Nenv ; env_index++){
						MatrixXdr env_all_zb = all_zb.array().colwise() * Enviro.col(env_index).array();
						output_env = compute_XXz(num_snp,env_all_zb);
						output_env = output_env.array().colwise() * Enviro.col(env_index).array();

						int gxe_bin_index;
						if(Annot_x_E == true)
							gxe_bin_index = Nbin + (env_index * Nbin) + bin_index;
						else
							gxe_bin_index = Nbin + env_index;

						for (int z_index = 0 ; z_index < Nz ; z_index++){
							XXz.col((gxe_bin_index * 2*Nz) + Nz + z_index)+=output_env.col(z_index);   /// save whole sample
							XXz.col((gxe_bin_index * 2*Nz) + z_index)+=output_env.col(z_index);

							if(both_side_cov == true) {
								vec1 = output_env.col(z_index);
								w1 = covariate.transpose() * vec1;
								w2 = Q * w1;
								w3 = covariate * w2;

								UXXz.col((gxe_bin_index * 2*Nz) + Nz + z_index)+=w3;
								UXXz.col((gxe_bin_index * 2*Nz) + z_index)+=w3;
							}
						}

						if (both_side_cov == true){
							MatrixXdr env_all_Uzb = all_Uzb.array().colwise() * Enviro.col(env_index).array();
							output_env = compute_XXz(num_snp,env_all_Uzb);
							output_env = output_env.array().colwise() * Enviro.col(env_index).array();

							for (int z_index = 0 ; z_index < Nz ; z_index++){
								XXUz.col((gxe_bin_index * 2*Nz) + Nz + z_index)+=output_env.col(z_index);   /// save whole sample
								XXUz.col((gxe_bin_index * 2*Nz) + z_index)+=output_env.col(z_index); 
							}
						}
						if(both_side_cov == true)
							scaled_pheno = new_pheno.array() * Enviro.col(env_index).array();
						else
							scaled_pheno= pheno.array() * Enviro.col(env_index).array();


						double temp_e_yxxy;
						temp_e_yxxy= compute_yXXy(num_snp,scaled_pheno);

						yXXy(gxe_bin_index,1)+=temp_e_yxxy;
						yXXy(gxe_bin_index,0)+=temp_e_yxxy;

					}
				}//end gxe computations

				if (block_index == num_blocks - 1){
					for (int z_index = 0 ; z_index < Nz ; z_index++){
						XXz.col((bin_index * 2 * Nz) + Nz + z_index) += output_XXz.col(bin_index * Nz + z_index);   /// save whole sample contribution

						if(both_side_cov == true) {
							vec1 = output_XXz.col(bin_index * Nz + z_index);
							w1 = covariate.transpose() * vec1;
							w2 = Q * w1;
							w3 = covariate * w2;
							UXXz.col((bin_index * 2 * Nz) + Nz + z_index) += w3;
							UXXz.col((bin_index * 2 * Nz) + z_index) += w3;
						}
					}
				}

				if (both_side_cov == true){
					output = compute_XXUz(num_snp); 
					for (int z_index = 0 ; z_index < Nz ; z_index++)
						output_XXUz.col(bin_index * Nz + z_index) = output_XXUz.col(bin_index * Nz + z_index) + output.col(z_index);

					if (opt1 && block_index == num_blocks - 1) { 
						MatrixXdr tmpoutput = output_XXUz.block (0, bin_index * Nz, output_XXUz.rows(), Nz);	

						int rows = static_cast<int>(tmpoutput.rows());
					    int cols = static_cast<int>(tmpoutput.cols());
						write_matrix (xsum_ofs, tmpoutput);
					}
					if (block_index == num_blocks - 1){
						for (int z_index = 0 ; z_index < Nz ; z_index++){
							XXUz.col((bin_index * 2 * Nz) + Nz + z_index) += output_XXUz.col(bin_index * Nz + z_index);   /// save whole sample
						}	 
					}
				}

				if (verbose >= 2)
					cout << "pheno(" << pheno.rows() << "," << pheno.cols () << ") " << pheno.sum() << endl;

				}

				if(both_side_cov == false) {
					temp_yXXy = compute_yXXy(num_snp, pheno);
					output_yXXy(bin_index, 0) += temp_yXXy;
				} else {
					temp_yXXy = compute_yVXXVy(num_snp);
					output_yXXy(bin_index, 0) += temp_yXXy;
				}				
				if (opt1 && block_index == num_blocks - 1) {
					temp_yXXy = output_yXXy(bin_index, 0);
					if (use_ysum)
						ysum_ofs.write((char *) (&temp_yXXy), sizeof(double));
					else
						jack_yXXy(bin_index, jack_index) = temp_yXXy;
				}
				if (block_index == num_blocks - 1) 
					yXXy(bin_index,1) += output_yXXy(bin_index, 0);


				if (verbose >= 2) { 
					cout << "Pass 1: " << jack_index << " " << block_index << " " << bin_index << "\tXXz(" << XXz.rows() <<"," << XXz.cols() << ") "<< XXz.sum() << endl;
					cout << "Pass 1: " << jack_index << " " << block_index << " " << bin_index << "\ttemp_yXXy : " << temp_yXXy << endl;
					cout << "Pass 1: " << jack_index <<" " << block_index << " " << bin_index << "\tyXXy(" << yXXy.rows() <<"," << yXXy.cols() << ") "<< yXXy.sum() << endl;
					if (yXXy.rows () > 1) {
						cout << "Pass 1: " << jack_index <<" " << block_index << " " << bin_index << "\toutput_yXXy " << output_yXXy(0,0) <<", " << output_yXXy(1,0) << endl;
						cout << "Pass 1: " << jack_index <<" " << block_index << " " << bin_index << "\tyXXy " << yXXy(0,1) <<", " << yXXy(1,1) << endl;
					}
				}

				mm.clean_up();
				if(use_mailman == true){
					delete[] sum_op;
					delete[] partialsums;
					delete[] yint_e;
					delete[] yint_m;
					for (int i  = 0 ; i < hsegsize; i++)
						delete[] y_m [i];
					delete[] y_m;

					for (int i  = 0 ; i < g.Nindv; i++)
						delete[] y_e[i];
					delete[] y_e;

					vector< vector<int> >().swap(g.p);
					vector< vector<int> >().swap(allgen_mail[bin_index].p);
					g.columnsum.clear();
					g.columnsum2.clear();
					g.columnmeans.clear();
					g.columnmeans2.clear();
					allgen_mail[bin_index].columnsum.clear();
					allgen_mail[bin_index].columnsum2.clear();
					allgen_mail[bin_index].columnmeans.clear();
					allgen_mail[bin_index].columnmeans2.clear();
				}
			} // loop over bins

		} // loop over read blocks

	}//end loop over jackknife blocks
	cout << "Finished reading and computing over all blocks" << endl;
	cout << endl;
	cout << endl;

	if (hetero_noise == true) {
		MatrixXdr hetro_all_Uzb;
		for (int env_index = 0 ; env_index < Nenv ; env_index++){
		/// add hetero env noise
			MatrixXdr hetro_all_zb = all_zb.array().colwise() * Enviro.col(env_index).array();
			hetro_all_zb = hetro_all_zb.array().colwise() * Enviro.col(env_index).array();

			if(both_side_cov == true){
				hetro_all_Uzb = all_Uzb.array().colwise() * Enviro.col(env_index).array();
				hetro_all_Uzb = hetro_all_Uzb.array().colwise() * Enviro.col(env_index).array();
			}

			int hetro_index;
			if(Annot_x_E == true)
				hetro_index = Nbin + (Nenv * Nbin) + env_index;
			else
				hetro_index = Nbin + Nenv + env_index;
			for (int z_index = 0 ; z_index < Nz ; z_index++){
				XXz.col(((hetro_index) * 2*Nz) + Nz + z_index) = hetro_all_zb.col(z_index);
				if(both_side_cov == true){
					vec1 = hetro_all_zb.col(z_index);
					w1 = covariate.transpose() * vec1;
					w2 = Q * w1;
					w3 = covariate * w2;
					UXXz.col(((hetro_index) * 2*Nz) + Nz + z_index) = w3;
					XXUz.col(((hetro_index) * 2*Nz) + Nz + z_index) = hetro_all_Uzb.col(z_index);
				}
			}

			MatrixXdr scaled_pheno;
			if(both_side_cov == true)
				scaled_pheno = new_pheno.array() * Enviro.col(env_index).array();
			else
				scaled_pheno= pheno.array() * Enviro.col(env_index).array();

			yXXy(hetro_index,1) = (scaled_pheno.array() * scaled_pheno.array()).sum();
			len.push_back(1);
		}
	}

	if (verbose == 1) {
		cout << "Size of bins :" << endl;
		if (hetero_noise == true) {
			for(int i = 0 ; i < Nbin + nongen_Nbin + Nenv ; i++)
				cout << "bin " << i<<" : " << len[i]<<endl;
		} else {
			for(int i = 0 ; i < Nbin + nongen_Nbin ; i++)
				cout << "bin " << i<<" : " << len[i]<<endl;
		}
	}

	cout << "Number of individuals without missing phenotype and enviroment = " << mask.sum() << endl;
	cout << endl;
	cout << endl;

	//handle when jackknife block does not include any SNPs from a bin//refill
	for (int bin_index = 0 ; bin_index < T_Nbin ; bin_index++){
		for (int z_index = 0 ; z_index < Nz ; z_index++){
			XXz.col((bin_index * 2 * Nz) + z_index) = XXz.col((bin_index * 2 * Nz) + Nz + z_index);
			if(both_side_cov == true){
				UXXz.col((bin_index * 2*Nz) + z_index) = UXXz.col((bin_index * 2 * Nz) + Nz + z_index);
				XXUz.col((bin_index * 2*Nz) + z_index) = XXUz.col((bin_index * 2 * Nz) + Nz + z_index);  
			}
		}
		yXXy(bin_index,0)= yXXy(bin_index,1);
	}

	if (gen_by_env) {
		const int full_idx = 0; // main additive bin
		for (int full_idx = 0; full_idx < gen_Nbin; ++full_idx) overwrite_additive_with_gamma_bin(full_idx);
		// overwrite_additive_with_gamma_bin(full_idx);
	}

	if (opt1)  {
		if (!use_summary_genotypes){
			string prefix = command_line_opts.OUTPUT_FILE_PATH;
			string wgxsum_path = prefix + ".wgxsum";
			wgxsum_ofs.open(wgxsum_path.c_str(), std::ios_base::out);
			for (int bin_index = 0 ; bin_index < T_Nbin ; bin_index++){
				MatrixXdr tmpoutput = XXz.block (0, bin_index * 2 * Nz + Nz, XXz.rows(), Nz);
				write_matrix (wgxsum_ofs, tmpoutput);
				
				if (both_side_cov) {
					tmpoutput = UXXz.block (0, bin_index * 2 * Nz + Nz, UXXz.rows(), Nz);
					write_matrix (wgxsum_ofs, tmpoutput);
					tmpoutput = XXUz.block (0, bin_index * 2 * Nz + Nz, XXUz.rows(), Nz);
					write_matrix (wgxsum_ofs, tmpoutput);
				}
			}
			wgxsum_ofs.close ();
		}

		if (!use_summary_genotypes)
			xsum_ofs.close();
		if (use_ysum)
			ysum_ofs.close();
	}
}

void genotype_stream_pass (string name, int pass_num){
	if (verbose >= 3) {
		cout << "both_side_cov = " << both_side_cov << endl;
		cout << "cov_add_intercept = " << cov_add_intercept << endl;
		cout << "T_Nbin = " << T_Nbin << " Nbin = " << Nbin << " nongen_Nbin = " << nongen_Nbin << endl;
		cout << "Nz = " << Nz << endl;
	}	
	if (opt1){ 
		string prefix = command_line_opts.OUTPUT_FILE_PATH;
		if (use_summary_genotypes) {
			xsum_path = xsumfilepath + ".xsum";
			wgxsum_path = xsumfilepath + ".wgxsum";
		} else
			xsum_path = prefix + ".xsum";
		ysum_path = prefix + ".ysum";
		if (pass_num==1) {
			if (!use_summary_genotypes)  
				xsum_ofs.open(xsum_path.c_str(), std::ios_base::out);
			if (use_ysum)	
				ysum_ofs.open(ysum_path.c_str(), std::ios_base::out);
		} else {
			xsum_ifs.open(xsum_path.c_str(), std::ios_base::in);
			if (!xsum_ifs.is_open()) {	
				cerr << "Error reading file "<< xsum_path <<endl;
				exit(1);
			}
			if (use_ysum)	
				ysum_ifs.open(ysum_path.c_str(), std::ios_base::in);
		}
	}


	ifstream ifs (name.c_str(), ios::in|ios::binary);
	if (!ifs.is_open()){
		cerr << "Error reading file "<< name <<endl;
		exit(1);
	}

	read_header = true;
	global_snp_index=-1;

	MatrixXdr output;
	MatrixXdr output_env;

	MatrixXdr vec1;
	MatrixXdr w1;
	MatrixXdr w2;
	MatrixXdr w3;

	MatrixXdr A_trs(T_Nbin,T_Nbin);
	MatrixXdr b_trk(T_Nbin,1);
	MatrixXdr c_yky(T_Nbin,1);

	MatrixXdr X_l(T_Nbin,T_Nbin);
	MatrixXdr Y_r(T_Nbin,1);
	MatrixXdr B1;
	MatrixXdr B2;
	MatrixXdr C1;
	MatrixXdr C2;
	double trkij;
	double yy = (pheno.array() * pheno.array()).sum();

	if(both_side_cov == true){
		yy = (new_pheno.array() * new_pheno.array()).sum();
	}

	Nindv_mask = mask.sum();
	if(both_side_cov == true)
		NC = Nindv_mask - Ncov;
	else
		NC = Nindv_mask;

	MatrixXdr herit;

	// Matrix of jacknife estimates
	// Rows: statistics (=variance components including sigma_e). Hence number of rows = T_Nbin + 1
	// Columns: jackknife estimates
	jack.resize(T_Nbin, Njack);

	// Matrix of variance components
	// Includes sigma_e. Hence number of entries = T_Nbin + 1
	point_est.resize(T_Nbin, 1);

	point_est_adj_gxe = MatrixXdr::Zero(T_Nbin,1);
	jack_adj_gxe = MatrixXdr::Zero(T_Nbin,Njack);

	enrich_jack.resize(T_Nbin, Njack);
	enrich_point_est.resize(T_Nbin, 1);


	MatrixXdr h1;
	MatrixXdr h2;
	MatrixXdr h3;

	double trkij_res1;
	double trkij_res2;
	double trkij_res3;
	double tk_res;


	if (pass_num == 2) {
		if (use_summary_genotypes){

			wgxsum_ifs.open(wgxsum_path.c_str(), std::ios_base::in);
			if (!wgxsum_ifs.is_open()) {	
				cerr << "Error reading file "<< wgxsum_path <<endl;
				exit(1);
			}

			for (int bin_index = 0 ; bin_index < T_Nbin ; bin_index++){
				read_matrix (wgxsum_ifs, output);
				XXz.block (0, bin_index * 2 * Nz + Nz, Nindv, Nz) = output;
				if (both_side_cov) {
					read_matrix (wgxsum_ifs, output);
					UXXz.block (0, bin_index * 2 * Nz + Nz, Nindv, Nz) = output;
					read_matrix (wgxsum_ifs, output);
					XXUz.block (0, bin_index * 2 * Nz + Nz, Nindv, Nz) = output;
				}
			}


			// Handle when a jackknife block does not include any SNPs from a bin
			// Only need to do this for genotype matrices (XXz, UXXz, XXUz) since
			// the phenotype summaries (yXXy) are being generated

			for (int bin_index = 0 ; bin_index < T_Nbin ; bin_index++){
				for (int z_index = 0 ; z_index < Nz ; z_index++){
					XXz.col((bin_index * 2*Nz) + z_index) = XXz.col((bin_index * 2*Nz) + Nz + z_index);
					if(both_side_cov == true){
						UXXz.col((bin_index * 2*Nz) + z_index) = UXXz.col((bin_index * 2*Nz) + Nz + z_index);
						XXUz.col((bin_index * 2*Nz) + z_index) = XXUz.col((bin_index * 2*Nz) + Nz + z_index);
					}
				}
//				yXXy(bin_index,0)= yXXy(bin_index,1);
			}

			wgxsum_ifs.close ();

			if (gen_by_env) {
				const int full_idx = 0;
				for (int full_idx = 0; full_idx < gen_Nbin; ++full_idx) overwrite_additive_with_gamma_bin(full_idx);
				// overwrite_additive_with_gamma_bin(full_idx);
			}
		}
	}

	for (int jack_index = 0 ; jack_index < Njack ; jack_index++){

		int read_Nsnp = jack_block_size[jack_index];	
		cout << "Pass "<< pass_num << ": Reading jackknife block " << jack_index << endl;
		
		if (verbose >= 1)  {
			cout << "************Pass " << pass_num << ": Reading jackknife block " << jack_index << " ************" <<endl;
			if (verbose >= 2)
			cout << "read_Nsnp = " << read_Nsnp << endl;
		}

		if(use_mailman == true){
			for (int i = 0 ; i < Nbin ; i++){
				allgen_mail[i].segment_size_hori = floor(log(Nindv) / log(3)) - 2 ;
				allgen_mail[i].Nsegments_hori = ceil(jack_bin[jack_index][i]*1.0 / (allgen_mail[i].segment_size_hori * 1.0));
				allgen_mail[i].p.resize(allgen_mail[i].Nsegments_hori,vector<int>(Nindv));

				allgen_mail[i].index = 0;
				allgen_mail[i].Nsnp = jack_bin[jack_index][i];
				allgen_mail[i].Nindv = Nindv;

				allgen_mail[i].columnsum.resize(jack_bin[jack_index][i],1);
				for (int index_temp = 0 ; index_temp < jack_bin[jack_index][i];index_temp++)
					allgen_mail[i].columnsum[index_temp]=0;
			}
		} else {
			for (int k = 0 ; k < Nbin ; k++){
				allgen[k].gen.resize(jack_bin[jack_index][k],Nindv);
				// number of SNPs in this annotation (bin) in this jackknife block
				allgen[k].index = 0;
			}
		}

		if (opt1  && pass_num == 2) {}
		else { 
			if(use_1col_annot == true)
				read_bed_1colannot(ifs, missing, read_Nsnp);
			else
				read_bed2(ifs, missing, read_Nsnp);
			read_header = false;
		}

		if (verbose >= 1) { 
			cout << "Number of SNPs in each bin in jackknife block "<< jack_index << endl;
			vectorfn::printvector (jack_bin[jack_index]); cout << endl;
		}

		for (int bin_index = 0 ; bin_index < Nbin ; bin_index++){
			int num_snp;
			if (opt1 && pass_num == 2) { 
				xsum_ifs.read((char *) (&num_snp), sizeof(int)); 
			} else { 
				if (use_mailman == true)
					num_snp = allgen_mail[bin_index].index;
				else
					num_snp = allgen[bin_index].index;
			}

			if (verbose >= 2)
				cout << "Number of SNPs in bin " << bin_index << "  = " << num_snp << endl; 
	
			// Skip empty bin
			if (num_snp == 0)  {
				if (opt1 && pass_num == 1) 
					xsum_ofs.write((char *) (&num_snp), sizeof(int));
				continue;
			}

			if (opt1 && pass_num==2) {
				read_matrix (xsum_ifs, output);
				
				if (verbose >= 2) {
					cout << "Pass "<< pass_num << ": " << jack_index << " " << bin_index << "\toutput(" << output.rows() <<"," << output.cols() << ") "<< output.sum() << endl;
				}
			} else {
				stds.resize(num_snp,1);
				means.resize(num_snp,1);

				if(use_mailman == true){
					for (int i = 0 ; i < num_snp ; i++)
						means(i,0) = (double)allgen_mail[bin_index].columnsum[i]/Nindv;
				} else	  
					means = allgen[bin_index].gen.rowwise().mean();


				for (int i = 0 ; i < num_snp ; i++)
					stds(i,0) = 1 / sqrt((means(i,0) * (1-(0.5 * means(i,0)))));

				if (use_mailman == true){
					g = allgen_mail[bin_index];
					g.segment_size_hori = floor(log(Nindv) / log(3)) - 2 ;
					g.Nsegments_hori = ceil(jack_bin[jack_index][bin_index]*1.0 / (g.segment_size_hori * 1.0));
					g.p.resize(g.Nsegments_hori,vector<int>(Nindv));
					initial_var();
				} else {
					gen = allgen[bin_index].gen;
				}

				mm = MatMult(g, gen, debug, var_normalize, memory_efficient, missing, use_mailman, nthreads, Nz);
				output = compute_XXz(num_snp, all_zb);

				if (verbose >= 2) 
					cout << "Pass "<< pass_num << ": " << jack_index << " " << bin_index << "\toutput(" << output.rows() <<"," << output.cols() << ") " << output.sum() << endl;

				if (opt1) {
					xsum_ofs.write((char *) (&num_snp), sizeof(int));
					write_matrix (xsum_ofs, output);
				}
			}

			// begin gxe computations
			// This code block has not been tested for the memory optimized setting
			MatrixXdr scaled_pheno;
			if (gen_by_env == true) { 
				// Not tested
				for (int env_index = 0 ; env_index < Nenv ; env_index++){
					MatrixXdr env_all_zb = all_zb.array().colwise() * Enviro.col(env_index).array();
					output_env = compute_XXz(num_snp,env_all_zb);
					output_env = output_env.array().colwise() * Enviro.col(env_index).array();

					int gxe_bin_index;
					if(Annot_x_E == true)
						gxe_bin_index = Nbin + (env_index * Nbin) + bin_index;
					else
						gxe_bin_index = Nbin + env_index;

					for (int z_index = 0 ; z_index < Nz ; z_index++){
						if(pass_num == 1){
							XXz.col((gxe_bin_index * 2*Nz) + Nz + z_index)+=output_env.col(z_index);   /// save whole sample
							XXz.col((gxe_bin_index * 2*Nz) + z_index)+=output_env.col(z_index);
						}
						else if(num_snp != len[bin_index])
							XXz.col((gxe_bin_index * 2*Nz) + z_index) = XXz.col((gxe_bin_index * 2*Nz) + z_index)-output_env.col(z_index);   /// save corresponding jack contrib

						if(both_side_cov == true) {
							vec1 = output_env.col(z_index);
							w1 = covariate.transpose() * vec1;
							w2 = Q * w1;
							w3 = covariate * w2;

							if(pass_num == 1){
								UXXz.col((gxe_bin_index * 2*Nz) + Nz + z_index)+=w3;
								UXXz.col((gxe_bin_index * 2*Nz) + z_index)+=w3;
							}
							else //if(num_snp != len[bin_index])
								UXXz.col((gxe_bin_index * 2*Nz) + z_index) = UXXz.col((gxe_bin_index * 2*Nz) + z_index)-w3;
						}
					}

					if (both_side_cov == true){
						MatrixXdr env_all_Uzb = all_Uzb.array().colwise() * Enviro.col(env_index).array();
						output_env = compute_XXz(num_snp,env_all_Uzb);
						output_env = output_env.array().colwise() * Enviro.col(env_index).array();

						for (int z_index = 0 ; z_index < Nz ; z_index++){
							if(pass_num == 1){
								XXUz.col((gxe_bin_index * 2*Nz) + Nz + z_index)+=output_env.col(z_index);   /// save whole sample
								XXUz.col((gxe_bin_index * 2*Nz) + z_index)+=output_env.col(z_index); 
							}
							else
								XXUz.col((gxe_bin_index * 2*Nz) + z_index) = XXUz.col((gxe_bin_index * 2*Nz) + z_index)-output_env.col(z_index);
						}
					}
					if(both_side_cov == true)
						scaled_pheno = new_pheno.array() * Enviro.col(env_index).array();
					else
						scaled_pheno= pheno.array() * Enviro.col(env_index).array();


					double temp_e_yxxy;
					temp_e_yxxy= compute_yXXy(num_snp,scaled_pheno);

					if(pass_num == 1){
						yXXy(gxe_bin_index,1)+=temp_e_yxxy;
						yXXy(gxe_bin_index,0)+=temp_e_yxxy;
					}
					else 
						yXXy(gxe_bin_index,0)= yXXy(gxe_bin_index,0)-temp_e_yxxy;      

				}
			}//end gxe computations

			// Number of columns = 2 * Number of annotations * Number of random vectors
			// For each annotation, the second half of XXz contains the whole genome statistic
			// The first half contains the jackknife statistic for the current jackknife block
			// These jackknife statistics are used to compute the jackknife variance components before moving onto the next jackknife block
			for (int z_index = 0 ; z_index < Nz ; z_index++){
				if(pass_num == 1){
					XXz.col((bin_index * 2 * Nz) + Nz + z_index) += output.col(z_index);   /// save whole genome contribution
				}
				else //if(num_snp != len[bin_index])
					XXz.col((bin_index * 2 * Nz) + z_index) = XXz.col((bin_index * 2 * Nz) + Nz + z_index)-output.col(z_index);   /// save corresponding jack contribution

				if(both_side_cov == true) {
					vec1 = output.col(z_index);
					w1 = covariate.transpose() * vec1;
					w2 = Q * w1;
					w3 = covariate * w2;
					if(pass_num == 1){
						UXXz.col((bin_index * 2*Nz) + Nz + z_index)+=w3;
						UXXz.col((bin_index * 2*Nz) + z_index)+=w3;
					}
					else if(num_snp != len[bin_index])
						UXXz.col((bin_index * 2*Nz) + z_index) = UXXz.col((bin_index * 2*Nz) + Nz + z_index)-w3;
				}
			}

			if (both_side_cov == true){
				if (opt1 && pass_num == 2) { 
					read_matrix (xsum_ifs, output);
				} else {
					output = compute_XXUz(num_snp); 
					if (opt1) {
						write_matrix (xsum_ofs, output);
					}
				}
				for (int z_index = 0 ; z_index < Nz ; z_index++){
					if(pass_num == 1){
						XXUz.col((bin_index * 2 * Nz) + Nz + z_index) += output.col(z_index);   /// save whole sample
					}
					else //if(num_snp != len[bin_index])
						XXUz.col((bin_index * 2 * Nz) + z_index) = XXUz.col((bin_index * 2*Nz) + Nz + z_index)-output.col(z_index);
				}	 
			}

			//compute yXXy
			double temp_yXXy;
			if (opt1 && pass_num == 2) { 
				if (use_ysum) 
					ysum_ifs.read((char *) (&temp_yXXy), sizeof(double)); 
				else
					temp_yXXy = jack_yXXy(bin_index, jack_index);
			} else {
				if (verbose >= 2)
					cout << "pheno(" << pheno.rows() << "," << pheno.cols () << ") " << pheno.sum() << endl;
				if(both_side_cov == false)
					temp_yXXy = compute_yXXy(num_snp, pheno);
				else
					temp_yXXy = compute_yVXXVy(num_snp);
					
				if (opt1) {
					jack_yXXy(bin_index, jack_index) = temp_yXXy;
					ysum_ofs.write((char *) (&temp_yXXy), sizeof(double));
				}	
			}

			// In the first pass, compute the whole-sample statistics: yXXy(,1)
			// In the second pass, compute the jackknife statistic for the current block
			// These jackknife statistics are used to compute the jackknife variance components before moving onto the next jackknife block
			if(pass_num == 1){
				yXXy(bin_index, 1) += temp_yXXy;
			}
			else 
				yXXy(bin_index, 0)= yXXy(bin_index , 1) - temp_yXXy;


			if (verbose >= 2) {
				cout << "Pass "<< pass_num << ": " << jack_index << " " << bin_index << "\tXXz(" << XXz.rows() <<"," << XXz.cols() << ") "<< XXz.sum() << endl;
				cout << "Pass "<< pass_num << ": " << jack_index << " " <<  bin_index << "\ttemp_yXXy: " << temp_yXXy << endl;
				cout << "Pass "<< pass_num << ": " << jack_index << " " << bin_index << "\tyXXy(" << yXXy.rows() <<"," << yXXy.cols() << ") "<< yXXy.sum() << endl;
				if (yXXy.rows () > 1)
					cout << "Pass " << pass_num << ": " << jack_index <<" " << bin_index << "\tyXXy " << yXXy(0,1) <<"," << yXXy(1,1) << endl;
			}

			if (opt1 && pass_num == 2){ 
			} else {
				mm.clean_up();
				if(use_mailman == true){

					delete[] sum_op;
					delete[] partialsums;
					delete[] yint_e;
					delete[] yint_m;
					for (int i  = 0 ; i < hsegsize; i++)
						delete[] y_m [i];
					delete[] y_m;

					for (int i  = 0 ; i < g.Nindv; i++)
						delete[] y_e[i];
					delete[] y_e;

					vector< vector<int> >().swap(g.p);
					vector< vector<int> >().swap(allgen_mail[bin_index].p);
					g.columnsum.clear();
					g.columnsum2.clear();
					g.columnmeans.clear();
					g.columnmeans2.clear();
					allgen_mail[bin_index].columnsum.clear();
					allgen_mail[bin_index].columnsum2.clear();
					allgen_mail[bin_index].columnmeans.clear();
					allgen_mail[bin_index].columnmeans2.clear();
				}
			}
		} // loop over bins

		if(pass_num == 2){
			// 	
			// Compute variance components for each jackknife subsample
			//
			for(int l = 0 ; l < T_Nbin ; l++)
				if( len[l]==jack_bin[jack_index][l])
					jack_bin[jack_index][l]=0;

			for (int i = 0 ; i < T_Nbin ; i++){
				b_trk(i,0) = Nindv_mask;

				if(i >= (T_Nbin-(nongen_Nbin + Nenv)) ){
					B1 = XXz.block(0, (i * 2 * Nz), Nindv, Nz);
					B1 =all_zb.array() * B1.array();
					b_trk(i,0) = B1.sum() / (len[i]-jack_bin[jack_index][i]) / Nz;
				}

				if (jack_index == Njack - 1){
					if (verbose >= 2)
						cout << "yXXy(bin,0) = " << yXXy(i,0) << ", Number of SNPs in bin[" <<i<<"] " << len[i] << " Number of SNPs in jackknife block[" << jack_index << "], bin[" <<i<<"] " << jack_bin[jack_index][i] << endl;
				}

				c_yky(i,0) = yXXy(i,0) / (len[i]-jack_bin[jack_index][i]);

				if(both_side_cov == true){
					B1 = XXz.block(0,(i * 2*Nz),Nindv,Nz);
					C1 = B1.array() * all_Uzb.array();
					C2 = C1.colwise().sum();	
					tk_res = C2.sum();  

					tk_res = tk_res / (len[i]-jack_bin[jack_index][i]) / Nz;

					b_trk(i,0) = b_trk(i,0)-tk_res;
				}

				for (int j = i ; j < T_Nbin ; j++){
					B1 = XXz.block(0,(i * 2*Nz),Nindv,Nz);
					B2 = XXz.block(0,(j * 2*Nz),Nindv,Nz);
					C1 = B1.array() * B2.array();
					C2 = C1.colwise().sum();
					trkij = C2.sum();


					if(both_side_cov == true){
						h1 = covariate.transpose() * B1;
						h2 = Q * h1;
						h3 = covariate * h2;
						C1 = h3.array() * B2.array();
						C2 = C1.colwise().sum();
						trkij_res1 = C2.sum();


						B1 = XXUz.block(0,(i * 2*Nz),Nindv,Nz);
						B2 = UXXz.block(0,(j * 2*Nz),Nindv,Nz);
						C1 = B1.array() * B2.array();
						C2 = C1.colwise().sum();
						trkij_res3 = C2.sum();


						trkij += trkij_res3 - trkij_res1 - trkij_res1 ;

					}

					trkij = trkij / (len[i]-jack_bin[jack_index][i]) / (len[j]-jack_bin[jack_index][j]) / Nz;
					A_trs(i,j) = trkij;
					A_trs(j,i) = trkij;

				}
			}


			// X_l << A_trs,b_trk,b_trk.transpose(),NC;
			// Y_r << c_yky,yy;

			X_l << A_trs;
			Y_r << c_yky;

			herit = X_l.colPivHouseholderQr().solve(Y_r);


			if(jack_index == 0){
				outfile << "Number of individuals after filtering = " << Nindv_mask << endl;
				outfile << "Number of covariates = " << Ncov << endl;
				outfile << "Number of environments = " << Nenv << endl;

				if (verbose >= 2) {
					cout << "Jackknife block = " << jack_index << endl;
					cout << "Xl[" << jack_index << "] = " << X_l << endl;
					cout << "Yr[" << jack_index << "] = " << Y_r << endl;
					double relative_error = (X_l * herit - Y_r).norm() / Y_r.norm(); // norm() is L2 norm
					cout << "The relative error is: " << relative_error << endl;
					
					#ifdef USE_DOUBLE
						JacobiSVD<MatrixXd> svd(X_l);
					#else
						JacobiSVD<MatrixXf> svd(X_l);
					#endif
					double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
					cout << "Condition number:  "<< cond << endl;

					outfile << "Jackknife block = " << jack_index << endl;
					outfile << "Xl[" << jack_index << "] = " << X_l << endl;
					outfile << "Yr[" << jack_index << "] = " << Y_r << endl;
					outfile << "Normal Equations info:" << endl;
					outfile << "The relative error is: " << relative_error << endl;
					outfile << "Max sing.val: " << svd.singularValues()(0) << endl;
					outfile << "Min sing.val: " << svd.singularValues()(svd.singularValues().size()-1) << endl;
					outfile << "Condition number: " << cond << endl;
					cout << endl;
					cout << endl;
					outfile << endl;
					outfile << endl;
				}
			}

			if (jack_index == 2){
				if (verbose >= 2){
					cout << "Jackknife block = " << jack_index << endl;
					cout << "Xl[" << jack_index << "] = " << X_l << endl;
					cout << "Yr[" << jack_index << "] = " << Y_r << endl;
				}
			}

			// Fill in the jackknife statistics
			for(int i = 0 ; i<(T_Nbin);i++)
				jack(i,jack_index) = herit(i,0);


			for(int i = 0 ; i<(T_Nbin);i++)
				// if(i == T_Nbin)
				// 	jack_adj_gxe(i,jack_index) = jack(i,jack_index) * NC;
				// else    
				jack_adj_gxe(i,jack_index) = jack(i,jack_index) * b_trk(i,0);


			//handle when a jackknife block does not include any SNPs from a bin

			for (int bin_index = 0 ; bin_index < T_Nbin ; bin_index++){
				for (int z_index = 0 ; z_index < Nz ; z_index++){
					XXz.col((bin_index * 2*Nz) + z_index) = XXz.col((bin_index * 2*Nz) + Nz + z_index);
					if(both_side_cov == true){
						UXXz.col((bin_index * 2*Nz) + z_index) = UXXz.col((bin_index * 2*Nz) + Nz + z_index);
						XXUz.col((bin_index * 2*Nz) + z_index) = XXUz.col((bin_index * 2*Nz) + Nz + z_index);
					}
				}
				yXXy(bin_index,0)= yXXy(bin_index,1);
			}

			if (trace){
				for (int i=0; i< Nbin; i++){
						for (int j=0; j < Nbin; j++){
								trace_file << (X_l(i, j) - Nindv_mask) * (len[i] - jack_bin[jack_index][i])*(len[j] - jack_bin[jack_index][j])/pow(Nindv_mask, 2)
								<< ",";
						}
						trace_file << len[i] - jack_bin[jack_index][i] << endl;
				}
			}

		} //end if pass_num = 2

	}//end loop over jackknife blocks
	cout << "Finished reading and computing over all blocks" << endl;
	cout << endl;
	cout << endl;

	if(pass_num == 1){
		if (hetero_noise == true) {
			MatrixXdr hetro_all_Uzb;
			for (int env_index = 0 ; env_index < Nenv ; env_index++){
			/// add hetero env noise
				MatrixXdr hetro_all_zb = all_zb.array().colwise() * Enviro.col(env_index).array();
				hetro_all_zb = hetro_all_zb.array().colwise() * Enviro.col(env_index).array();

				if(both_side_cov == true){
					hetro_all_Uzb = all_Uzb.array().colwise() * Enviro.col(env_index).array();
					hetro_all_Uzb = hetro_all_Uzb.array().colwise() * Enviro.col(env_index).array();
				}

				int hetro_index;
				if(Annot_x_E == true)
					hetro_index = Nbin + (Nenv * Nbin) + env_index;
				else
					hetro_index = Nbin + Nenv + env_index;
				for (int z_index = 0 ; z_index < Nz ; z_index++){
					XXz.col(((hetro_index) * 2*Nz) + Nz + z_index) = hetro_all_zb.col(z_index);
					if(both_side_cov == true){
						vec1 = hetro_all_zb.col(z_index);
						w1 = covariate.transpose() * vec1;
						w2 = Q * w1;
						w3 = covariate * w2;
						UXXz.col(((hetro_index) * 2*Nz) + Nz + z_index) = w3;
						XXUz.col(((hetro_index) * 2*Nz) + Nz + z_index) = hetro_all_Uzb.col(z_index);
					}
				}

				MatrixXdr scaled_pheno;
				if(both_side_cov == true)
					scaled_pheno = new_pheno.array() * Enviro.col(env_index).array();
				else
					scaled_pheno= pheno.array() * Enviro.col(env_index).array();

				yXXy(hetro_index,1) = (scaled_pheno.array() * scaled_pheno.array()).sum();
				len.push_back(1);
			}
		}

		if (verbose >= 1) { 
			cout << "Size of bins :" << endl;
			if (hetero_noise == true) {
				for(int i = 0 ; i < Nbin + nongen_Nbin + Nenv ; i++)
					cout << "bin " << i<<" : " << len[i]<<endl;
			} else {
				for(int i = 0 ; i < Nbin + nongen_Nbin ; i++)
					cout << "bin " << i<<" : " << len[i]<<endl;
			}
		}

		cout << "Number of individuals without missing phenotype and enviroment: " << mask.sum() << endl;
		cout << endl;
		cout << endl;



		//handle when jackknife block does not include any SNPs from a bin//refill
		for (int bin_index = 0 ; bin_index < T_Nbin ; bin_index++){
			for (int z_index = 0 ; z_index < Nz ; z_index++){
				XXz.col((bin_index * 2*Nz) + z_index) = XXz.col((bin_index * 2*Nz) + Nz + z_index);
				if(both_side_cov == true){
					UXXz.col((bin_index * 2*Nz) + z_index) = UXXz.col((bin_index * 2*Nz) + Nz + z_index);
					XXUz.col((bin_index * 2*Nz) + z_index) = XXUz.col((bin_index * 2*Nz) + Nz + z_index);  
				}
			}
			yXXy(bin_index,0)= yXXy(bin_index,1);
		}

	} // pass 1

	if(pass_num == 2) {
		//
		// Compute variance components for the full sample
		//
		for (int i = 0 ; i < T_Nbin ; i++){
			c_yky(i,0) = yXXy(i,1) / len[i];
			//if(both_side_cov == false)
			b_trk(i,0) = Nindv_mask;

			if(i>=(T_Nbin-(nongen_Nbin + Nenv)) ){
				B1 = XXz.block(0,(i * 2*Nz) + Nz,Nindv,Nz);
				B1 =all_zb.array() * B1.array();
				b_trk(i,0) = B1.sum() / len[i]/Nz;
			}

			if(both_side_cov == true){
				B1 = XXz.block(0,(i * 2*Nz) + Nz,Nindv,Nz);
				C1 = B1.array() * all_Uzb.array();
				C2 = C1.colwise().sum();
				tk_res = C2.sum();
				tk_res = tk_res / len[i]/Nz;

				b_trk(i,0) = b_trk(i,0)-tk_res;
			}

			for (int j = i ; j < T_Nbin ; j++){
				B1 = XXz.block(0,(i * 2*Nz) + Nz,Nindv,Nz);
				B2 = XXz.block(0,(j * 2*Nz) + Nz,Nindv,Nz);
				C1 = B1.array() * B2.array();
				C2 = C1.colwise().sum();
				trkij = C2.sum();

				if(both_side_cov == true){
					h1 = covariate.transpose() * B1;
					h2 = Q * h1;
					h3 = covariate * h2;
					C1 = h3.array() * B2.array();
					C2 = C1.colwise().sum();
					trkij_res1 = C2.sum();
					B1 = XXUz.block(0,(i * 2*Nz) + Nz,Nindv,Nz);
					B2 = UXXz.block(0,(j * 2*Nz) + Nz,Nindv,Nz);
					C1 = B1.array() * B2.array();
					C2 = C1.colwise().sum();
					trkij_res3 = C2.sum();
					trkij += trkij_res3 - trkij_res1 - trkij_res1 ;

				}
				trkij = trkij / len[i]/len[j]/Nz;
				A_trs(i,j) = trkij;
				A_trs(j,i) = trkij;
			}
		} // loop over bins     


		// X_l << A_trs,b_trk,b_trk.transpose(),NC;
		// Y_r << c_yky,yy;

		X_l << A_trs;
		Y_r << c_yky;

		if (trace){
			for (int i=0; i< Nbin; i++){
					for (int j=0; j < Nbin; j++){
							trace_file << (X_l(i, j) - Nindv_mask) * len[i]*len[j]/pow(Nindv_mask,2) << ",";
					}
					trace_file << len[i] << endl;
			}
		}

		herit = X_l.colPivHouseholderQr().solve(Y_r);

		if (verbose >= 2 ) { 
			cout << "Whole-genome normal equations" << endl;
			cout << "Xl" << endl << X_l << endl;
			cout << "Yr" << endl << Y_r << endl;
		}
		for(int i = 0 ; i<(T_Nbin);i++)
			point_est(i,0) = herit(i,0);

		//point_est_adj_gxe = MatrixXdr::Zero(T_Nbin + 3,1);
		//jack_adj_gxe = MatrixXdr::Zero(T_Nbin + 3,Njack);

		for(int i = 0 ; i<(T_Nbin);i++)
			if(i == T_Nbin)
				point_est_adj_gxe(i,0) = point_est(i,0) * NC;
			else
				point_est_adj_gxe(i,0) = point_est(i,0) * b_trk(i,0);
	} // pass 2


	if (opt1){ 
		if (pass_num==1)  {
			if (!use_summary_genotypes) 
				xsum_ofs.close();
			if (use_ysum) 
				ysum_ofs.close();
		} else {
			xsum_ifs.close();
			if (!keep_xsum)
				std::remove (xsum_path.c_str());
			if (use_ysum)  {
				ysum_ifs.close();
				std::remove (ysum_path.c_str());
			}
		}
	}
}


/* This is the key code in ge_hetero_flexible.cpp*/
void genotype_stream_single_pass (string name) {
	if (verbose >= 3) {
		cout << "both_side_cov = " << both_side_cov << endl;
		cout << "cov_add_intercept = " << cov_add_intercept << endl;
		cout << "T_Nbin = " << T_Nbin << " Nbin = " << Nbin << " nongen_Nbin = " << nongen_Nbin << endl;
		cout << "Nz = " << Nz << endl;
	}

	ifstream ifs (name.c_str(), ios::in|ios::binary);
	if (!ifs.is_open()){
		cerr << "Error reading file "<< name <<endl;
		exit(1);
	}

	read_header = true;
	global_snp_index=-1;

	MatrixXdr output;
	MatrixXdr output_env;

	MatrixXdr vec1;
	MatrixXdr w1;
	MatrixXdr w2;
	MatrixXdr w3;

	for (int jack_index = 0; jack_index < Njack; jack_index ++){
		int read_Nsnp = jack_block_size[jack_index];	
		cout << "Reading jackknife block " << jack_index << endl;
		if (verbose >= 1)  {
			cout << "************Reading jackknife block " << jack_index << " ************" <<endl;
			if (verbose >= 2)
				cout << "read_Nsnp = " << read_Nsnp << endl;
		}


		//		int read_Nsnp = (jack_index<(Njack-1)) ? (step_size) : (step_size+step_size_rem);
		if(use_mailman == true){
			for (int i = 0; i < Nbin; i++){
				allgen_mail[i].segment_size_hori = floor(log(Nindv)/log(3)) - 2 ;
				allgen_mail[i].Nsegments_hori = ceil(jack_bin[jack_index][i]*1.0/(allgen_mail[i].segment_size_hori*1.0));
				allgen_mail[i].p.resize(allgen_mail[i].Nsegments_hori,std::vector<int>(Nindv));
				allgen_mail[i].not_O_i.resize(jack_bin[jack_index][i]);
				allgen_mail[i].not_O_j.resize(Nindv);
				allgen_mail[i].index = 0;
				allgen_mail[i].Nsnp = jack_bin[jack_index][i];
				allgen_mail[i].Nindv = Nindv;

				allgen_mail[i].columnsum.resize(jack_bin[jack_index][i],1);
				for (int index_temp = 0; index_temp < jack_bin[jack_index][i]; index_temp++)
					allgen_mail[i].columnsum[index_temp] = 0;
			}
		}
		else{
			for (int bin_index = 0;bin_index < Nbin; bin_index++){
				allgen[bin_index].gen.resize(jack_bin[jack_index][bin_index],Nindv);
				allgen[bin_index].index = 0;
			}
		}

		if(use_1col_annot == true)
			read_bed_1colannot (ifs, missing, read_Nsnp);
		else
			read_bed2 (ifs, missing, read_Nsnp);
		read_header = false;

		for (int bin_index = 0; bin_index < Nbin; bin_index++){
			int num_snp;
			if (use_mailman == true)
				num_snp = allgen_mail[bin_index].index;
			else
				num_snp = allgen[bin_index].index;

			if(num_snp != 0){
				stds.resize(num_snp, 1);
				means.resize(num_snp, 1);

				if(use_mailman == true){
					for (int i = 0; i < num_snp; i++)
						means(i,0) = (double)allgen_mail[bin_index].columnsum[i]/Nindv;
				}			
				else	  
					means = allgen[bin_index].gen.rowwise().mean();


				for (int i = 0; i < num_snp; i++)
					stds(i,0) = 1/sqrt((means(i,0)*(1-(0.5*means(i,0)))));

				if (use_mailman == true){
					g = allgen_mail[bin_index];
					g.segment_size_hori = floor(log(Nindv)/log(3)) - 2 ;
					g.Nsegments_hori = ceil(jack_bin[jack_index][bin_index]*1.0/(g.segment_size_hori*1.0));
					g.p.resize(g.Nsegments_hori,std::vector<int>(Nindv));
					g.not_O_i.resize(jack_bin[jack_index][bin_index]);
					g.not_O_j.resize(Nindv);
					initial_var();

				}else{
					gen = allgen[bin_index].gen;

				} 
				mm = MatMult(g, gen, debug, var_normalize, memory_efficient, missing, use_mailman, nthreads, Nz);
				// cout << "here1" << endl;
				output = compute_XXz(num_snp,all_zb);
				// cout << "here2" << endl;

				///gxe computations
				MatrixXdr scaled_pheno;
				scaled_pheno.resize(Nindv, phenocount);
				if (gen_by_env == true) {
					MatrixXdr temp;
					for (int env_index = 0; env_index < Nenv; env_index++){
						MatrixXdr env_all_zb = all_zb.array().colwise()*Enviro.col(env_index).array();                                   
						output_env = compute_XXz (num_snp, env_all_zb);                                   
						output_env = output_env.array().colwise()*Enviro.col(env_index).array();

						int gxe_bin_index;
						if(Annot_x_E == true)  
							gxe_bin_index = Nbin+(env_index*Nbin)+bin_index;
						else
							gxe_bin_index = Nbin+env_index;			

						for (int z_index = 0; z_index < Nz; z_index++){
							XXz.col(((gxe_bin_index)*(Njack+1)*Nz)+(jack_index*Nz)+z_index) += output_env.col(z_index);	
							XXz.col(((gxe_bin_index)*(Njack+1)*Nz)+(Njack*Nz)+z_index) += output_env.col(z_index);

							if(both_side_cov == true) {
								vec1 = output_env.col(z_index);
								w1 = covariate.transpose() * vec1;
								w2 = Q * w1;
								w3 = covariate * w2;
								//if(num_snp!=len[bin_index])
								UXXz.col(((gxe_bin_index)*(Njack+1)*Nz)+(jack_index*Nz)+z_index) += w3;
								UXXz.col(((gxe_bin_index)*(Njack+1)*Nz)+(Njack*Nz)+z_index) += w3;
							}

						}


						if (both_side_cov==true){
							MatrixXdr env_all_Uzb = all_Uzb.array().colwise()*Enviro.col(env_index).array();
							output_env = compute_XXz(num_snp,env_all_Uzb);
							output_env = output_env.array().colwise()*Enviro.col(env_index).array();

							for (int z_index = 0; z_index < Nz; z_index++){
								XXUz.col(((gxe_bin_index)*(Njack+1)*Nz)+(jack_index*Nz)+z_index) += output_env.col(z_index);
								XXUz.col(((gxe_bin_index)*(Njack+1)*Nz)+(Njack*Nz)+z_index) += output_env.col(z_index);   /// save whole sample
							}
						}
						for (int i = 0; i < phenocount; i++) {
							if(both_side_cov==true)
								scaled_pheno.col(i) = new_pheno.col(i).array()*Enviro.col(env_index).array();
							else
								scaled_pheno.col(i) = pheno.col(i).array()*Enviro.col(env_index).array();
						}

						if (phenocount > Nz) {
							int num_pheno_block = phenocount / Nz;
							int num_pheno_remain = phenocount % Nz;

							if (num_pheno_remain > 0) num_pheno_block += 1;

							int tstart = 0;
							int cur_num_pheno;
							for (int k = 0; k < num_pheno_block; k++) {
								if ((k == num_pheno_block - 1) && (num_pheno_remain > 0))
									cur_num_pheno = num_pheno_remain;
								else
									cur_num_pheno = Nz;
								MatrixXdr cur_scaled_pheno = scaled_pheno.block(0, tstart, scaled_pheno.rows(), cur_num_pheno);
								temp = compute_yXXy_multi(num_snp, cur_scaled_pheno, cur_num_pheno);
								yXXy.block(gxe_bin_index*phenocount+tstart, jack_index, cur_num_pheno, 1) += temp.transpose();
								yXXy.block(gxe_bin_index*phenocount+tstart, Njack, cur_num_pheno, 1) += temp.transpose();
								tstart += cur_num_pheno;
							}
						} else {
							temp = compute_yXXy_multi(num_snp, scaled_pheno, phenocount);
							yXXy.block(gxe_bin_index*phenocount, jack_index, phenocount, 1) += temp.transpose();
							yXXy.block(gxe_bin_index*phenocount, Njack, phenocount, 1) += temp.transpose();
						}
					}
				}
				////end gxe computation

				for (int z_index = 0; z_index < Nz; z_index++){
					if(num_snp != len[bin_index])
						XXz.col((bin_index*(Njack+1)*Nz)+(jack_index*Nz)+z_index) = output.col(z_index);
					XXz.col((bin_index*(Njack+1)*Nz)+(Njack*Nz)+z_index) += output.col(z_index);   /// save whole sample

					if(both_side_cov == true) {
						vec1 = output.col(z_index);
						w1 = covariate.transpose()*vec1;
						w2 = Q * w1;
						w3 = covariate * w2;
						if(num_snp != len[bin_index])
							UXXz.col((bin_index*(Njack+1)*Nz)+(jack_index*Nz)+z_index) = w3;
						UXXz.col((bin_index*(Njack+1)*Nz)+(Njack*Nz)+z_index) += w3;
					}

				}

				if (both_side_cov == true){
					output=compute_XXUz(num_snp); 
					for (int z_index = 0; z_index < Nz; z_index++){
						if(num_snp != len[bin_index])
							XXUz.col((bin_index*(Njack+1)*Nz)+(jack_index*Nz)+z_index) = output.col(z_index);
						XXUz.col((bin_index*(Njack+1)*Nz)+(Njack*Nz)+z_index) += output.col(z_index);   /// save whole sample
					}
				}

				//compute yXXy
				MatrixXdr temp_yxxy;
				if(both_side_cov==false){
						if (phenocount > Nz) {
								int num_pheno_block = phenocount / Nz;
								int num_pheno_remain = phenocount % Nz;

								if (num_pheno_remain > 0) num_pheno_block += 1;
								// if (jack_index == 0)
								//         cout << "Processing phenotypes in " << num_pheno_block << " blocks..." << endl;

								int tstart = 0;
								int cur_num_pheno;

								for (int k = 0; k < num_pheno_block; k++) {
										if ((k == num_pheno_block - 1) && (num_pheno_remain > 0))
												cur_num_pheno = num_pheno_remain;
										else
												cur_num_pheno = Nz;
										MatrixXdr cur_pheno = pheno.block(0, tstart, pheno.rows(), cur_num_pheno);
										temp_yxxy = compute_yXXy_multi(num_snp, cur_pheno, cur_num_pheno);


										if(num_snp!=len[bin_index]) {
												yXXy.block(bin_index*phenocount+tstart,jack_index,cur_num_pheno,1) += temp_yxxy.transpose();
										}
										yXXy.block(bin_index*phenocount+tstart,Njack,cur_num_pheno,1)+=temp_yxxy.transpose();
										tstart += cur_num_pheno;
								}

						} else {
								temp_yxxy=compute_yXXy_multi(num_snp, pheno, phenocount);	

								if(num_snp!=len[bin_index])
										yXXy.block(bin_index*phenocount,jack_index,phenocount,1)=temp_yxxy.transpose();
								yXXy.block(bin_index*phenocount,Njack,phenocount,1)+=temp_yxxy.transpose();
						}
						

				} else {
						if (phenocount > Nz) {
								int num_pheno_block = phenocount / Nz;
								int num_pheno_remain = phenocount % Nz;

								if (num_pheno_remain > 0) num_pheno_block += 1;
								// if (jack_index == 0)
								//         cout << "Processing phenotypes in " << num_pheno_block << " blocks..." << endl;

								int tstart = 0;
								int cur_num_pheno;
								for (int k = 0; k < num_pheno_block; k++) {
										if ((k == num_pheno_block - 1) && (num_pheno_remain > 0))
												cur_num_pheno = num_pheno_remain;
										else
												cur_num_pheno = Nz;
										MatrixXdr cur_pheno = new_pheno.block(0, tstart, new_pheno.rows(), cur_num_pheno);
										temp_yxxy = compute_yVXXVy_multi(num_snp, cur_pheno, cur_num_pheno);
										if(num_snp!=len[bin_index]) {
												yXXy.block(bin_index*phenocount+tstart,jack_index,cur_num_pheno,1) += temp_yxxy.transpose();
										}
										yXXy.block(bin_index*phenocount+tstart,Njack,cur_num_pheno,1)+=temp_yxxy.transpose();
										tstart += cur_num_pheno;
								}

						} else {
								temp_yxxy=compute_yVXXVy_multi(num_snp, new_pheno, phenocount);
								if(num_snp!=len[bin_index])
										yXXy.block(bin_index*phenocount,jack_index,phenocount,1)=temp_yxxy.transpose();
								yXXy.block(bin_index*phenocount,Njack,phenocount,1)+=temp_yxxy.transpose();
						}

				}




				if (verbose >= 2) {
					cout << jack_index << " " << bin_index << "\tXXz(" << XXz.rows() <<"," << XXz.cols() << ") "<< XXz.sum() << endl;
					cout << jack_index << " " << bin_index << "\tyXXy(" << yXXy.rows() <<"," << yXXy.cols() << ") "<< yXXy.sum() << endl;
					if (yXXy.rows () > 1)
						cout << jack_index <<" " << bin_index << "\tyXXy " << yXXy(0,1) <<"," << yXXy(1,1) << endl;
				}

				//compute Xz

				if (verbose >= 2) {
					cout << num_snp << " SNPs in bin "<< bin_index<< " of jackknife block  " << jack_index << endl;   
					cout << "Reading and computing bin " << bin_index << "  of jackknife block " << jack_index << " completed" << endl;
				}
				mm.clean_up();
				if(use_mailman==true){
					delete[] sum_op;
					delete[] partialsums;
					delete[] yint_e;
					delete[] yint_m;
					for (int i  = 0 ; i < hsegsize; i++)
						delete[] y_m [i];
					delete[] y_m;

					for (int i  = 0 ; i < g.Nindv; i++)
						delete[] y_e[i];
					delete[] y_e;

					std::vector< std::vector<int> >().swap(g.p);
					std::vector< std::vector<int> >().swap(g.not_O_j);
					std::vector< std::vector<int> >().swap(g.not_O_i);
					std::vector< std::vector<int> >().swap(allgen_mail[bin_index].p);
					std::vector< std::vector<int> >().swap(allgen_mail[bin_index].not_O_j);
					std::vector< std::vector<int> >().swap(allgen_mail[bin_index].not_O_i);
					g.columnsum.clear();
					g.columnsum2.clear();
					g.columnmeans.clear();
					g.columnmeans2.clear();
					allgen_mail[bin_index].columnsum.clear();
					allgen_mail[bin_index].columnsum2.clear();
					allgen_mail[bin_index].columnmeans.clear();
					allgen_mail[bin_index].columnmeans2.clear();
				}
			}
		} // loop over bins
	}//end loop over jackknife blocks
	cout << "Finished reading and computing over all blocks" << endl;
	cout << endl;
	cout << endl;


	if (hetero_noise == true) {
		MatrixXdr hetro_all_Uzb;
		for (int env_index=0;env_index<Nenv;env_index++){
			/// add hetero env noise
			MatrixXdr hetro_all_zb=all_zb.array().colwise()*Enviro.col(env_index).array();
			hetro_all_zb=hetro_all_zb.array().colwise()*Enviro.col(env_index).array();

			if(both_side_cov==true){
				hetro_all_Uzb=all_Uzb.array().colwise()*Enviro.col(env_index).array();
				hetro_all_Uzb=hetro_all_Uzb.array().colwise()*Enviro.col(env_index).array();
			}

			int hetro_index;
			if(Annot_x_E==true)
				hetro_index=Nbin+(Nenv*Nbin)+env_index;
			else
				hetro_index=Nbin+Nenv+env_index;
			for (int z_index=0;z_index<Nz;z_index++){
				XXz.col(((hetro_index)*(Njack+1)*Nz)+(Njack*Nz)+z_index)=hetro_all_zb.col(z_index);		 
				if(both_side_cov==true){
					vec1=hetro_all_zb.col(z_index);
					w1=covariate.transpose()*vec1;
					w2=Q*w1;
					w3=covariate*w2;
					UXXz.col(((hetro_index)*(Njack+1)*Nz)+(Njack*Nz)+z_index)=w3;
					XXUz.col(((hetro_index)*(Njack+1)*Nz)+(Njack*Nz)+z_index)=hetro_all_Uzb.col(z_index);
				}

			}

			MatrixXdr scaled_pheno;
			scaled_pheno.resize(Nindv, phenocount);
			for (int i = 0; i < phenocount; i++) {
					if(both_side_cov==true)
							scaled_pheno.col(i) = new_pheno.col(i).array()*Enviro.col(env_index).array();
					else
							scaled_pheno.col(i) = pheno.col(i).array()*Enviro.col(env_index).array();
					yXXy(hetro_index*phenocount+i, Njack)= (scaled_pheno.col(i).array()*scaled_pheno.col(i).array()).sum();
			}
			len.push_back(1);
		} 
	}

	cout<<"Size of bins :"<<endl;
	//CHANGE(10/20)
	if (hetero_noise == true) {
		for(int i=0;i<Nbin+nongen_Nbin+Nenv;i++)
			cout<<"bin "<<i<<" : "<<len[i]<<endl;
	} else {
		for(int i=0;i<Nbin+nongen_Nbin;i++)
			cout<<"bin "<<i<<" : "<<len[i]<<endl;
	}

	for (int phen_index = 0; phen_index < phenocount; phen_index++) {
			cout<<"Number of individuals after filtering for phenotype " << phen_index << " : "<<mask.col(phen_index).sum()<<endl;
	}
	cout << endl;
	cout << endl;
	cout << endl;
	gen_Nbin=Nbin;

	//CHANGE(10/20)
	if (hetero_noise == true)
		Nbin = Nbin + nongen_Nbin + Nenv;
	else
		Nbin = Nbin + nongen_Nbin;
	T_Nbin = Nbin;
	
	for(int bin_index = 0; bin_index < Nbin; bin_index++){
		for(int jack_index = 0; jack_index < Njack; jack_index++){
			for (int z_index = 0; z_index < Nz; z_index++){
				MatrixXdr v1 = XXz.col((bin_index*(Njack+1)*Nz)+(Njack*Nz)+z_index);
				MatrixXdr v2 = XXz.col((bin_index*(Njack+1)*Nz)+(jack_index*Nz)+z_index);
				XXz.col((bin_index*(Njack+1)*Nz)+(jack_index*Nz)+z_index) = v1-v2;
				if(both_side_cov == true){
					v1 = XXUz.col((bin_index*(Njack+1)*Nz)+(Njack*Nz)+z_index);
					v2 = XXUz.col((bin_index*(Njack+1)*Nz)+(jack_index*Nz)+z_index);
					XXUz.col((bin_index*(Njack+1)*Nz)+(jack_index*Nz)+z_index) = v1-v2;

					v1 = UXXz.col((bin_index*(Njack+1)*Nz)+(Njack*Nz)+z_index);
					v2 = UXXz.col((bin_index*(Njack+1)*Nz)+(jack_index*Nz)+z_index);
					UXXz.col((bin_index*(Njack+1)*Nz)+(jack_index*Nz)+z_index) = v1-v2;
				}    
			}
			// yXXy(bin_index,jack_index) = yXXy(bin_index,Njack)-yXXy(bin_index,jack_index);
			yXXy.block(bin_index*phenocount,jack_index,phenocount,1)=yXXy.block(bin_index*phenocount,Njack,phenocount,1)-yXXy.block(bin_index*phenocount,jack_index,phenocount,1);
		}
	}


	if (gen_by_env) {
		const int full_idx = 0; // the additive bin you want to repurpose for gamma
		for (int full_idx = 0; full_idx < gen_Nbin; ++full_idx) overwrite_additive_with_gamma_bin(full_idx);
		// overwrite_additive_with_gamma_bin(full_idx);
	}

	//// all XXy and yXXy and contributions of every jackknife subsamples  were computed till this line.

	/// normal equations LHS
	MatrixXdr  A_trs(Nbin,Nbin);
	MatrixXdr b_trk(Nbin,1);
	MatrixXdr c_yky(Nbin,1);

	MatrixXdr X_l(Nbin,Nbin);
	MatrixXdr Y_r(Nbin,1);

	int jack_index=Njack;
	MatrixXdr B1;
	MatrixXdr B2;
	MatrixXdr C1;
	MatrixXdr C2;
	double trkij;
	// double yy = (pheno.array() * pheno.array()).sum();

	// if(both_side_cov == true)
	// 	yy = (new_pheno.array()*new_pheno.array()).sum();

	MatrixXdr all_yy;
	if(both_side_cov == true)
			all_yy=(new_pheno.array() * new_pheno.array()).colwise().sum();
	else
			all_yy=(pheno.array() * pheno.array()).colwise().sum();

	int Nindv_mask = mask.sum();
	if(both_side_cov == true)
		NC = Nindv_mask - Ncov;
	else
		NC = Nindv_mask;


	/*
	MatrixXdr jack;
	MatrixXdr point_est;
	MatrixXdr enrich_jack;
	MatrixXdr enrich_point_est;

	//// adjust for GxE
	MatrixXdr point_est_adj_gxe;
	MatrixXdr jack_adj_gxe;
*/

	point_est_adj_gxe=MatrixXdr::Zero(Nbin,phenocount);
	jack_adj_gxe=MatrixXdr::Zero(phenocount*(Nbin),Njack);


	jack.resize(phenocount*(Nbin),Njack);
	point_est.resize(Nbin,phenocount);

	enrich_jack.resize(phenocount*Nbin,Njack);
	enrich_point_est.resize(Nbin,phenocount);

	MatrixXdr h1;
	MatrixXdr h2;
	MatrixXdr h3;

	double trkij_res1;
	double trkij_res2;
	double trkij_res3;
	double tk_res;



	for (jack_index=0;jack_index<=Njack;jack_index++){
		for(int k=0;k<Nbin;k++)
			if( jack_index<Njack && len[k]==jack_bin[jack_index][k])
				jack_bin[jack_index][k]=0;

		for (int i = 0; i < Nbin; i++) {
			for (int j = i; j < Nbin; j++) {
				B1=XXz.block(0,(i*(Njack+1)*Nz)+(jack_index*Nz),Nindv,Nz);
				B2=XXz.block(0,(j*(Njack+1)*Nz)+(jack_index*Nz),Nindv,Nz);
				C1=B1.array()*B2.array();
				C2=C1.colwise().sum();
				trkij=C2.sum();

				if(both_side_cov==true){

					h1=covariate.transpose()*B1;
					h2=Q*h1;
					h3=covariate*h2;
					C1=h3.array()*B2.array();
					C2=C1.colwise().sum();
					trkij_res1=C2.sum();

					B1=XXUz.block(0,(i*(Njack+1)*Nz)+(jack_index*Nz),Nindv,Nz);
					B2=UXXz.block(0,(j*(Njack+1)*Nz)+(jack_index*Nz),Nindv,Nz);
					C1=B1.array()*B2.array();
					C2=C1.colwise().sum();
					trkij_res3=C2.sum();

					trkij+=trkij_res3-trkij_res1-trkij_res1 ;
					

				}
				if(jack_index==Njack) {
					trkij=trkij/len[i]/len[j]/Nz;
				} else {
					trkij=trkij/(len[i]-jack_bin[jack_index][i])/(len[j]-jack_bin[jack_index][j])/Nz;
				}
					
				A_trs(i,j)=trkij;
				A_trs(j,i)=trkij;

			}
		}

		for (int k = 0; k < phenocount; k++) {
			for (int i = 0; i < Nbin; i++) {
				b_trk(i,0)=mask.col(k).sum();
				// CHANGE (2/17)
				int sum_num_nongen_bin = 0;
				if (hetero_noise == true) 
					sum_num_nongen_bin = nongen_Nbin + Nenv;
				else 
					sum_num_nongen_bin = nongen_Nbin;

				if(i>=(Nbin-sum_num_nongen_bin) ){

					B1=XXz.block(0,(i*(Njack+1)*Nz)+(jack_index*Nz),Nindv,Nz);
					B1 =all_zb.array()*B1.array();

					if(jack_index==Njack)
						b_trk(i,0)=B1.sum()/len[i]/Nz;
					else
						b_trk(i,0)=B1.sum()/(len[i]-jack_bin[jack_index][i])/Nz;

				}

				if(jack_index==Njack)
					c_yky(i,0)=yXXy((i*phenocount)+k,jack_index)/len[i];
				else
					c_yky(i,0)=yXXy((i*phenocount)+k,jack_index)/(len[i]-jack_bin[jack_index][i]);


				if(both_side_cov==true){
					B1=XXz.block(0,(i*(Njack+1)*Nz)+(jack_index*Nz),Nindv,Nz);
					C1=B1.array()*all_Uzb.array();
					C2=C1.colwise().sum();	
					tk_res=C2.sum();  

					if(jack_index==Njack)
						tk_res=tk_res/len[i]/Nz;
					else
						tk_res=tk_res/(len[i]-jack_bin[jack_index][i])/Nz;

					b_trk(i,0)=b_trk(i,0)-tk_res;
				}
			}
			
			if (both_side_cov == true)
				NC = mask.col(k).sum() - Ncov;
			else
				NC = mask.col(k).sum();

			double yy = all_yy(0, k);

			// X_l<<A_trs,b_trk,b_trk.transpose(),NC;
			// Y_r<<c_yky,yy;
	
			X_l << A_trs;
			Y_r << c_yky;


			double cur_nindv_mask = mask.col(k).sum();
			if (trace){
				if (jack_index < Njack){
					for (int i=0; i< Nbin; i++){
						for (int j=0; j < Nbin; j++){
							trace_file << (X_l(i, j) - cur_nindv_mask) * (len[i] - jack_bin[jack_index][i])*(len[j] - jack_bin[jack_index][j])/pow(cur_nindv_mask, 2)
								<< ",";
						}
						trace_file << len[i] - jack_bin[jack_index][i] << endl;
					}
				}
				else{
					for (int i=0; i< Nbin; i++){
						for (int j=0; j < Nbin; j++){
							trace_file << (X_l(i, j) - cur_nindv_mask) * len[i]*len[j]/pow(cur_nindv_mask,2) << ",";
						}
						trace_file << len[i] << endl;
					}

				}
			}

			MatrixXdr herit = X_l.fullPivHouseholderQr().solve(Y_r);


			if(jack_index == Njack){
				if (k == 0) {
					outfile << "Number of covariates: " << Ncov << endl;
					outfile << "Number of environments: " << Nenv << endl;
				}

				outfile << "Number of individuals after filtering for phenotype " <<  k << " : " << cur_nindv_mask << endl;
				if (verbose == true) {
					cout << "Phenotype " << k << ": " << endl;
					cout<<"LHS of Normal Eq"<<endl<<X_l<<endl;
					cout<<"RHS of Normal Eq"<<endl<<Y_r<<endl;
					double relative_error = (X_l*herit - Y_r).norm() / Y_r.norm(); // norm() is L2 norm
					cout << "The relative error is: " << relative_error << endl;
					
					#ifdef USE_DOUBLE
									JacobiSVD<MatrixXd> svd(X_l);
					#else
									JacobiSVD<MatrixXf> svd(X_l);
					#endif

					double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
					cout<<"condition number:  "<< cond<<endl;

					outfile<<"LHS of Normal Eq"<<endl<<X_l<<endl;
					outfile<<"RHS of Normal Eq"<<endl<<Y_r<<endl;
					outfile<<"Normal Equations info:"<<endl;
					outfile<<"The relative error is: " << relative_error << endl;
					outfile<<"Max sing.val: "<<svd.singularValues()(0)<<endl;
					outfile<<"Min sing.val: "<<svd.singularValues()(svd.singularValues().size()-1)<<endl;
					outfile<<"Condition number: "<<cond<<endl;
					cout << endl;
					cout << endl;
					outfile << endl;
					outfile << endl;
				}
				for(int i = 0; i < (Nbin); i++)
					point_est(i,k) = herit(i,0);

				//adj gxe

				for(int i=0;i<(Nbin);i++){
					// if(i==Nbin)
					// 	point_est_adj_gxe(i,k)=point_est(i,k)*NC;
					// else
					point_est_adj_gxe(i,k)=point_est(i,k)*b_trk(i,0);
				} 
			}else{
				for(int i=0;i<(Nbin);i++)
						jack(k*(Nbin)+i,jack_index)=herit(i,0);

				//adj gxe
				for(int i=0;i<(Nbin);i++){
						// if(i==Nbin)
						// 		jack_adj_gxe(k*(Nbin+3)+i,jack_index)=jack(k*(Nbin+1)+i,jack_index)*NC;
						// else
						jack_adj_gxe(k*(Nbin)+i,jack_index)=jack(k*(Nbin)+i,jack_index)*b_trk(i,0);
				}
			}
		}
	}//end of loop over jack

}



// Regress covariates from phenotypes
//  
void regress_covariates () {
	bool normalize_proj_pheno = command_line_opts.normalize_proj_pheno;
	#ifdef USE_DOUBLE	
		Eigen::VectorXd sums = pheno.colwise().sum();
	#else
		Eigen::VectorXf sums = pheno.colwise().sum();
	#endif
	double single_mean;
	if(use_cov == true){
		MatrixXdr mat_mask = mask.col(0).replicate(1,Ncov);
		covariate = covariate.cwiseProduct(mat_mask);
		MatrixXdr WtW= covariate.transpose() * covariate;
		Q = WtW.inverse(); // Q = (W^tW)^-1
		MatrixXdr temp_cov;
		if (both_side_cov == false){
			for (int i = 0; i < phenocount; i++) {
				mat_mask=mask.col(i).replicate(1,Ncov);
				temp_cov=covariate.cwiseProduct(mat_mask); 
				WtW=temp_cov.transpose()*temp_cov;
				Q=WtW.inverse(); // Q=(W^tW)^-1
				v1=temp_cov.transpose()*pheno.col(i);
				v2=Q*v1;
				v3=temp_cov*v2;
				new_pheno.col(i)=pheno.col(i)-v3;
				new_pheno.col(i)=new_pheno.col(i).cwiseProduct(mask.col(i));

			}
		}

		if (both_side_cov == true){
			for (int i=0;i<phenocount;i++){
					single_mean=sums(i)/mask.col(i).sum();
					// cout<<mask.col(i).sum()<<endl;
					// cout<<"mean of phenotype"<<i<<": "<<single_mean<<endl;
					double phen_sd = 0;
					for(int j=0; j<Nindv; j++){
							phen_sd+=(pheno(j,i)-single_mean)*(pheno(j,i)-single_mean);
							if(mask(j,i)==1)
									pheno(j,i)=pheno(j,i)-single_mean; //center phenotype
					}
					phen_sd = sqrt(phen_sd/(mask.col(i).sum()-1));
					for (int j = 0; j < Nindv; j++) {
							if (mask(j, i)==1)
									pheno(j, i) = pheno(j, i)/phen_sd;
					}
			}

			for (int i = 0; i < phenocount; i++) {
					mat_mask=mask.col(i).replicate(1,Ncov);
					temp_cov=covariate.cwiseProduct(mat_mask); 
					WtW=temp_cov.transpose()*temp_cov;
					Q=WtW.inverse(); // Q=(W^tW)^-1
					v1=temp_cov.transpose()*pheno.col(i);
					v2=Q*v1;
					v3=temp_cov*v2;
					new_pheno.col(i)=pheno.col(i)-v3;
					new_pheno.col(i)=new_pheno.col(i).cwiseProduct(mask.col(i));
			}
			/// centering
			if (normalize_proj_pheno == true) {
					for (int i=0;i<phenocount;i++){
							single_mean=new_pheno.col(i).sum()/mask.col(i).sum();
							double phen_sd = 0;
							for(int j=0; j<Nindv; j++){
									phen_sd+=(new_pheno(j,i)-single_mean)*(new_pheno(j,i)-single_mean);
									if(mask(j,i)==1)
											new_pheno(j,i)=new_pheno(j,i)-single_mean;
							}
							phen_sd = sqrt(phen_sd/(mask.col(i).sum()-1));
							for (int j = 0; j < Nindv; j++) {
									if (mask(j, i)==1)
											new_pheno(j, i) = new_pheno(j, i)/phen_sd;
							}
					}
			}

		}

	} else {
		for (int i=0;i<phenocount;i++){
				single_mean=sums(i)/mask.col(i).sum();
				// cout<<mask.col(i).sum()<<endl;
				// cout<<"mean of phenotype"<<i<<": "<<single_mean<<endl;
				for(int j=0; j<Nindv; j++){
						if(mask(j,i)==1)
								pheno(j,i)=pheno(j,i)-single_mean; //center phenotype
				}
		}

	}
}

// Assumes: pheno, mask, phenocount, Nindv, boxcox_lambdas are global.
void apply_boxcox_transform() {
    if (boxcox_lambdas.empty()) return;

    const int Nind = pheno.rows();
    const int P    = pheno.cols();
    const int L    = static_cast<int>(boxcox_lambdas.size());

    // New matrices: each original phenotype p gets L transformed versions
    MatrixXdr pheno_bc(Nind, P * L);
    MatrixXdr mask_bc (Nind, P * L);

    for (int p = 0; p < P; ++p) {
        for (int l = 0; l < L; ++l) {
            const double lambda = boxcox_lambdas[l];
            const int col = p * L + l;

            // Global stats across all non-missing samples
            double sum   = 0.0;
            double sumsq = 0.0;
            int n        = 0;

            // --- First pass: compute global mean and sd ---
            for (int i = 0; i < Nind; ++i) {
                if (mask(i, p) == 0) continue; // skip missing phenotype

                const double y = pheno(i, p);

                // Box-Cox requires strictly positive y
                if (y <= 0.0) {
                    std::cerr << "ERROR: Box-Cox requires positive phenotype values. "
                              << "Found y <= 0 for individual " << i
                              << ", phenotype " << p << std::endl;
                    exit(1);
                }

                double z;
                if (std::fabs(lambda) < 1e-8) {
                    z = std::log(y);
                } else {
                    z = (std::pow(y, lambda) - 1.0) / lambda;
                }

                sum   += z;
                sumsq += z * z;
                ++n;
            }

            double mean = 0.0;
            double sd   = 1.0;

            if (n == 0) {
                std::cerr << "WARNING: phenotype " << p
                          << ", lambda index " << l
                          << " has no non-missing values.\n";
            } else {
                mean = sum / n;
                double var = (sumsq / n) - mean * mean;
                sd = (var > 0.0) ? std::sqrt(var) : 1.0;
            }

            // --- Second pass: write globally standardized transformed values ---
            for (int i = 0; i < Nind; ++i) {
                if (mask(i, p) == 0) {
                    // preserve missingness
                    pheno_bc(i, col) = 0.0;
                    mask_bc (i, col) = 0.0;
                    continue;
                }

                const double y = pheno(i, p);
                double z;
                if (std::fabs(lambda) < 1e-8) {
                    z = std::log(y);
                } else {
                    z = (std::pow(y, lambda) - 1.0) / lambda;
                }

                const double z_std = (z - mean) / sd;
                pheno_bc(i, col) = z_std;
                mask_bc (i, col) = 1.0;
            }
        }
    }

    // Replace originals with Box-Cox transformed versions
    pheno.swap(pheno_bc);
    mask.swap(mask_bc);
    phenocount = pheno.cols();  // now P * L

    // Recompute y_sum if needed
    y_sum = pheno.col(0).sum();
}


// Read .bim, .pheno, .env, .fam, .annot, .cov file.
void read_auxillary_files () { 

	// Read bim file to count number of SNPs
	string geno_name = command_line_opts.GENOTYPE_FILE_PATH;
	std::stringstream f1;
	f1 << geno_name << ".bim";

	if (jack_scheme == 1|| jack_scheme == 2){ 
		Nsnp = get_number_of_snps (f1.str());
		step_size = Nsnp / Njack;
		step_size_rem = Nsnp%Njack;
		read_bim (f1.str());
	} else {
		Nsnp = read_bim (f1.str());
	}

	setup_read_blocks();

	// trace summary only
	use_dummy = command_line_opts.use_dummy_pheno;

	// Read .fam file
	std::stringstream f0;
	f0 << geno_name << ".fam";
	string name_fam = f0.str();
	int fam_lines = count_fam (name_fam);
	
	// Read phenotype and save the number of indvs
	string filename = command_line_opts.PHENOTYPE_FILE_PATH;

	if (use_dummy){
		Nindv = fam_lines;
	}	
	else{
		Nindv = count_pheno (filename);
		read_pheno (Nindv, filename);
	}

	if (fam_lines != Nindv) {
		exitWithError ("Number of individuals in fam file and pheno file does not match ");
		exit(1);
	}

	cout << "Number of individuals = "<< Nindv << endl;

	// Read .env file depending on the model
	gen_by_env = command_line_opts.gen_by_env;
	hetero_noise = command_line_opts.hetero_noise;
	if (gen_by_env == false) {
		Nenv = 0;
		hetero_noise = false;
	} else {
		std::string envfile = command_line_opts.ENV_FILE_PATH;
		Nenv = read_env_bin(Nindv, envfile);
	}

	// Build lambda grid if requested
	setup_boxcox_lambdas();

	// Apply Box-Cox transform AFTER env is loaded
	if (command_line_opts.use_boxcox && !boxcox_lambdas.empty()) {
		if (!gen_by_env) {
			std::cerr << "ERROR: environment-specific Box-Cox normalization requires "
			          << "--gen_by_env / env file to be provided." << std::endl;
			exit(1);
		}

		apply_boxcox_transform();

		new_pheno.resize(Nindv, phenocount);
		new_pheno.setZero();
	}

	y_sum = pheno.sum();

	// Read annotation files
	filename = command_line_opts.Annot_PATH;

	if(use_1col_annot == true){
		read_annot_1col(filename);
	}else{
		read_annot(filename);
	}

	// Read covariate file
	std::string covfile = command_line_opts.COVARIATE_FILE_PATH;
	std::string covname = "";
	if(covfile != "" ){
		use_cov = true;
		Ncov = read_cov (Nindv, covfile);
	} else if (covfile == ""){
		cout << "No covariate file specified" << endl;

		if ((cov_add_intercept == true) && (!use_dummy)) {
			covariate.resize(Nindv,1);
			for(int i = 0 ; i < Nindv ; i++)
				covariate(i,0) = 1;
			Ncov = 1;
			use_cov = true;
			cout << "Intercept included" << endl;
		} else {
			both_side_cov = false;
			use_cov = false;
			cout << "No intercept included" << endl;
		}
	}

	xsumfilepath = command_line_opts.XSUM_FILE_PATH;
	if(xsumfilepath != "" ){
		use_summary_genotypes = true;	
		string xsum_path = xsumfilepath + ".xsum";
		string wgxsum_path = xsumfilepath + ".wgxsum";
		xsum_ifs.open(xsum_path.c_str(), std::ios_base::in);
		if (!xsum_ifs.is_open()) {	
			cerr << "Error reading file "<< xsum_path <<endl;
			exit(1);
		}
		xsum_ifs.close();

		wgxsum_ifs.open(wgxsum_path.c_str(), std::ios_base::in);
		if (!wgxsum_ifs.is_open()) {	
			cerr << "Error reading file "<< wgxsum_path <<endl;
			exit(1);
		}
		wgxsum_ifs.close();
	}	
}


void print_results () {
    int T_Nbin;
    if (gen_by_env == false) {
        T_Nbin = gen_Nbin;
    } else if (hetero_noise == true) {
        T_Nbin = gen_Nbin + nongen_Nbin + Nenv;
    } else {
        T_Nbin = gen_Nbin + nongen_Nbin;
    }

    // Jackknife SEs for *raw* variance components (before any rescaling)
    MatrixXdr point_se = statsfn::jack_se(jack);

    // Flat storage for rho and rho SE
    std::vector<double> rho_hat_flat;
    std::vector<double> rho_se_flat;
    bool store_rho = (Nenv > 1 && gen_Nbin > 0 && phenocount > 0);

    auto alloc_rho = [&]() {
        if (!store_rho) return;
        std::size_t total =
            (std::size_t)phenocount *
            (std::size_t)gen_Nbin *
            (std::size_t)Nenv *
            (std::size_t)Nenv;
        rho_hat_flat.assign(total, std::numeric_limits<double>::quiet_NaN());
        rho_se_flat.assign(total, std::numeric_limits<double>::quiet_NaN());
    };

    auto rho_index = [&](int phen, int bin, int e1, int e2) -> std::size_t {
        return ((((std::size_t)phen * (std::size_t)gen_Nbin + (std::size_t)bin)
                 * (std::size_t)Nenv + (std::size_t)e1)
                 * (std::size_t)Nenv + (std::size_t)e2);
    };

    if (store_rho) {
        alloc_rho();
    }

    // Basic info
    cout << "*****" << endl;
    outfile << "*****" << endl;
    for (int i = 0 ; i < T_Nbin ; i++){
        cout << "Number of SNPs in bin " << i << " = " << len[i] << endl;
        outfile << "Number of SNPs in bin " << i << " = " << len[i] << endl;
    }
    cout << "*****" << endl;
    outfile << "*****" << endl;
    cout << "Number of G variance components = " << gen_Nbin << endl;
    cout << "Number of GxE variance components = " << nongen_Nbin << endl;
    outfile << "Number of G variance components = " << gen_Nbin << endl;
    outfile << "Number of GxE variance components = " << nongen_Nbin << endl;
    if (hetero_noise == true) {
        cout << "Number of NxE variance components = " << Nenv << endl;
        outfile << "Number of NxE variance components = " << Nenv << endl;
    } else {
        cout << "Number of NxE variance components = 0" << endl;
        outfile << "Number of NxE variance components = 0" << endl;
    }

    cout << "*****" << endl;
    outfile << "*****" << endl;

    if (use_dummy){
        cout << "!!! A dummy phenotype is used for this GENIE run. "
             << "The heritability estimates are NOT meaningful "
             << "(please use the trace summaries only) !!!" << endl;
        cout << "*****" << endl;
        outfile << "!!! A dummy phenotype is used for this GENIE run. "
                << "The heritability estimates are NOT meaningful "
                << "(please use the trace summaries only) !!!" << endl;
        outfile << "*****" << endl;
    }

    // Preserve the raw estimates before we renormalise anything
    MatrixXdr point_est_preserved = point_est;

    cout << endl << "OUTPUT: " << endl;
    outfile << endl << "OUTPUT: " << endl;
    if (verbose >= 3) {
        cout << "gen_Nbin = " << gen_Nbin << "\tT_Nbin = " << T_Nbin << endl;
    }

    // ================= RAW VARIANCE COMPONENTS PER PHENOTYPE =================
    for (int phen_index = 0; phen_index < phenocount; ++phen_index) {
        cout << endl << "Phenotype " << phen_index << ":" << endl;
        outfile << endl << "Phenotype " << phen_index << ":" << endl;

        cout << "Variance components:" << endl;
        outfile << "Variance components:" << endl;

        for (int j = 0 ; j < T_Nbin ; ++j) {
            int se_row = phen_index * (T_Nbin) + j;

            if (j < gen_Nbin) {
                // Gamma (annotation main effects)
                cout << "Gamma[bin=" << j << "] : "
                     << point_est_preserved(j, phen_index)
                     << "  SE : " << point_se(se_row, 0) << endl;
                outfile << "Gamma[bin=" << j << "] : "
                        << point_est_preserved(j, phen_index)
                        << "  SE : " << point_se(se_row, 0) << endl;
            }
            else if (j < (gen_Nbin + nongen_Nbin)) {
                // GxE terms, env-major layout
                int k = j - gen_Nbin;
                int env_index = (gen_Nbin > 0 ? k / gen_Nbin : 0);
                int bin_index = (gen_Nbin > 0 ? k % gen_Nbin : 0);

                cout << "Sigma^2_g[env=" << env_index
                     << "][bin=" << bin_index << "] : "
                     << point_est_preserved(j, phen_index)
                     << "  SE : " << point_se(se_row, 0) << endl;

                outfile << "Sigma^2_g[env=" << env_index
                        << "][bin=" << bin_index << "] : "
                        << point_est_preserved(j, phen_index)
                        << "  SE : " << point_se(se_row, 0) << endl;
            }
            else {
                // NxE terms (pure env variance components)
                int env_index = j - gen_Nbin - nongen_Nbin;

                cout << "Sigma^2_e[env=" << env_index << "] : "
                     << point_est_preserved(j, phen_index)
                     << "  SE : " << point_se(se_row, 0) << endl;

                outfile << "Sigma^2_e[env=" << env_index << "] : "
                        << point_est_preserved(j, phen_index)
                        << "  SE : " << point_se(se_row, 0) << endl;
            }
        }
    }

    // ================= INDEXING HELPERS (ENV-MAJOR LAYOUT) =================
    const int nb_annot   = gen_Nbin;                           // # annotation bins
    const int nb_gxe_tot = nongen_Nbin;                        // total GxE bins
    const int nb_nxe_tot = T_Nbin - (gen_Nbin + nongen_Nbin);  // total NxE bins

    const int off_gxe = nb_annot;                  // start of GxE block
    const int off_nxe = nb_annot + nb_gxe_tot;     // start of NxE block

    const int stride_gxe = (Nenv > 0 ? nb_gxe_tot / Nenv : 0); // # GxE bins per env
    const int stride_nxe = (Nenv > 0 ? nb_nxe_tot / Nenv : 0); // # NxE bins per env

    auto gxe_idx = [&](int e, int a){
        return off_gxe + e * stride_gxe + a;
    };
    auto nxe_idx = [&](int e, int b){
        return off_nxe + e * stride_nxe + b;
    };

    // ================= 1) ENV-WISE NORMALISATION: point_est =================
    for (int phen_index = 0; phen_index < phenocount; ++phen_index) {
        if (Nenv <= 0) continue;

        std::vector<double> denom(Nenv, 0.0);

        for (int e = 0; e < Nenv; ++e) {
            for (int a = 0; a < stride_gxe; ++a)
                denom[e] += point_est(gxe_idx(e, a), phen_index);
            for (int b = 0; b < stride_nxe; ++b)
                denom[e] += point_est(nxe_idx(e, b), phen_index);
        }

        for (int e = 0; e < Nenv; ++e) {
            double d = denom[e];
            if (d <= 0.0) continue;
            for (int a = 0; a < stride_gxe; ++a)
                point_est(gxe_idx(e, a), phen_index) /= d;
            for (int b = 0; b < stride_nxe; ++b)
                point_est(nxe_idx(e, b), phen_index) /= d;
        }
    }

    // ================= 2) ENV-WISE NORMALISATION: jack =================
    for (int phen_index = 0; phen_index < phenocount; ++phen_index) {
        if (Nenv <= 0) continue;

        MatrixXdr block = jack.block(phen_index * (T_Nbin), 0,
                                     T_Nbin, Njack);

        for (int i = 0; i < Njack; ++i) {
            std::vector<double> denom(Nenv, 0.0);

            for (int e = 0; e < Nenv; ++e) {
                for (int a = 0; a < stride_gxe; ++a)
                    denom[e] += block(gxe_idx(e, a), i);
                for (int b = 0; b < stride_nxe; ++b)
                    denom[e] += block(nxe_idx(e, b), i);
            }

            for (int e = 0; e < Nenv; ++e) {
                double d = denom[e];
                if (d <= 0.0) continue;
                for (int a = 0; a < stride_gxe; ++a)
                    block(gxe_idx(e, a), i) /= d;
                for (int b = 0; b < stride_nxe; ++b)
                    block(nxe_idx(e, b), i) /= d;
            }
        }

        jack.block(phen_index * (T_Nbin), 0,
                   T_Nbin, Njack) = block;
    }

    // ================= 3) ENV-WISE NORMALISATION: point_est_adj_gxe =================
    for (int phen_index = 0; phen_index < phenocount; ++phen_index) {
        if (Nenv <= 0) continue;

        std::vector<double> denom(Nenv, 0.0);

        for (int e = 0; e < Nenv; ++e) {
            for (int a = 0; a < stride_gxe; ++a)
                denom[e] += point_est_adj_gxe(gxe_idx(e, a), phen_index);
            for (int b = 0; b < stride_nxe; ++b)
                denom[e] += point_est_adj_gxe(nxe_idx(e, b), phen_index);
        }

        for (int e = 0; e < Nenv; ++e) {
            double d = denom[e];
            if (d <= 0.0) continue;
            for (int a = 0; a < stride_gxe; ++a)
                point_est_adj_gxe(gxe_idx(e, a), phen_index) /= d;
            for (int b = 0; b < stride_nxe; ++b)
                point_est_adj_gxe(nxe_idx(e, b), phen_index) /= d;
        }
    }

    // ================= 4) ENV-WISE NORMALISATION: jack_adj_gxe =================
    for (int phen_index = 0; phen_index < phenocount; ++phen_index) {
        if (Nenv <= 0) continue;

        // jack_adj_gxe has (T_Nbin+3) rows per phenotype; we only rescale 0..T_Nbin-1
        MatrixXdr block = jack_adj_gxe.block(phen_index * (T_Nbin), 0,
                                             T_Nbin, Njack);

        for (int i = 0; i < Njack; ++i) {
            std::vector<double> denom(Nenv, 0.0);

            for (int e = 0; e < Nenv; ++e) {
                for (int a = 0; a < stride_gxe; ++a)
                    denom[e] += block(gxe_idx(e, a), i);
                for (int b = 0; b < stride_nxe; ++b)
                    denom[e] += block(nxe_idx(e, b), i);
            }

            for (int e = 0; e < Nenv; ++e) {
                double d = denom[e];
                if (d <= 0.0) continue;
                for (int a = 0; a < stride_gxe; ++a)
                    block(gxe_idx(e, a), i) /= d;
                for (int b = 0; b < stride_nxe; ++b)
                    block(nxe_idx(e, b), i) /= d;
            }
        }

        jack_adj_gxe.block(phen_index * (T_Nbin), 0,
                           T_Nbin, Njack) = block;
    }

    // SEs for env-wise normalised heritabilities
    MatrixXdr SEjack_adj_gxe = statsfn::jack_se(jack_adj_gxe);

    // ================= SUMMARIES: rho(bin, env1-env2) =================
    cout << "*****" << endl;
    outfile << "*****" << endl;
    cout << "Summaries: " << endl;
    outfile << "Summaries: " << endl;

    for (int phen_index = 0; phen_index < phenocount; ++phen_index) {
        cout << "Phenotype " << phen_index << " (environment correlations):" << endl;
        outfile << "Phenotype " << phen_index << " (environment correlations):" << endl;

        MatrixXdr block = jack.block(phen_index * (T_Nbin), 0,
                                     T_Nbin, Njack);

        for (int bin = 0; bin < gen_Nbin; ++bin) {
            double gamma_val = point_est(bin, phen_index);

            for (int e1 = 0; e1 < Nenv; ++e1) {
                int idx1 = gxe_idx(e1, bin);
                double sigma1 = point_est(idx1, phen_index);

                for (int e2 = e1 + 1; e2 < Nenv; ++e2) {
                    int idx2 = gxe_idx(e2, bin);
                    double sigma2 = point_est(idx2, phen_index);

                    double rho_hat = 0.0;
                    double rho_se  = 0.0;
                    bool valid = (sigma1 > 0.0 && sigma2 > 0.0);

                    if (valid) {
                        rho_hat = gamma_val / std::sqrt(sigma1 * sigma2);

                        std::vector<double> rho_vals;
                        rho_vals.reserve(Njack);

                        for (int i = 0; i < Njack; ++i) {
                            double gamma_i  = block(bin,  i);
                            double sigma1_i = block(idx1, i);
                            double sigma2_i = block(idx2, i);

                            if (sigma1_i > 0.0 && sigma2_i > 0.0) {
                                double rho_i = gamma_i / std::sqrt(sigma1_i * sigma2_i);
                                rho_vals.push_back(rho_i);
                            }
                        }

                        int m = static_cast<int>(rho_vals.size());
                        if (m > 1) {
                            double mean_rho = 0.0;
                            for (double v : rho_vals) mean_rho += v;
                            mean_rho /= m;

                            double ss = 0.0;
                            for (double v : rho_vals) {
                                double d = v - mean_rho;
                                ss += d * d;
                            }
                            rho_se = std::sqrt((double)(m - 1) / m * ss);
                        }

                        // store rho for wide table
                        if (store_rho) {
                            std::size_t idx_flat_1 = rho_index(phen_index, bin, e1, e2);
                            std::size_t idx_flat_2 = rho_index(phen_index, bin, e2, e1);
                            if (idx_flat_1 < rho_hat_flat.size()) {
                                rho_hat_flat[idx_flat_1] = rho_hat;
                                rho_se_flat [idx_flat_1] = rho_se;
                            }
                            if (idx_flat_2 < rho_hat_flat.size()) {
                                rho_hat_flat[idx_flat_2] = rho_hat;
                                rho_se_flat [idx_flat_2] = rho_se;
                            }
                        }
                    }

                    if (valid) {
                        cout << "rho[bin=" << bin
                             << ", env=" << e1 << "-" << e2 << "] : "
                             << rho_hat << "  SE : " << rho_se << endl;

                        outfile << "rho[bin=" << bin
                                << ", env=" << e1 << "-" << e2 << "] : "
                                << rho_hat << "  SE : " << rho_se << endl;
                    } else {
                        cout << "rho[bin=" << bin
                             << ", env=" << e1 << "-" << e2 << "] : NA" << endl;

                        outfile << "rho[bin=" << bin
                                << ", env=" << e1 << "-" << e2 << "] : NA" << endl;
                    }
                }
            }
        }
    }

    // ================= HERITABILITIES (ENV-WISE) =================
    cout << "*****" << endl;
    outfile << "*****" << endl;
    cout << "Heritabilities:" << endl;
    outfile << "Heritabilities:" << endl;

    for (int phen_index = 0; phen_index < phenocount; ++phen_index) {
        cout << "Phenotype " << phen_index << ":" << endl;
        outfile << "Phenotype " << phen_index << ":" << endl;

        for (int j = 0; j < T_Nbin; ++j) {
            if (j < gen_Nbin) continue; // skip Gamma bins for h2

            int se_row = phen_index * (T_Nbin) + j;

            if (j < (gen_Nbin + nongen_Nbin)) {
                int k = j - gen_Nbin;
                int env_index = (gen_Nbin > 0 ? k / gen_Nbin : 0);
                int bin_index = (gen_Nbin > 0 ? k % gen_Nbin : 0);

                cout << "h2_g[env=" << env_index
                     << "][bin=" << bin_index << "] : "
                     << point_est_adj_gxe(j, phen_index)
                     << "  SE : " << SEjack_adj_gxe(se_row, 0) << endl;

                outfile << "h2_g[env=" << env_index
                        << "][bin=" << bin_index << "] : "
                        << point_est_adj_gxe(j, phen_index)
                        << "  SE : " << SEjack_adj_gxe(se_row, 0) << endl;
            }
            else {
                int env_index = j - gen_Nbin - nongen_Nbin;

                cout << "h2_e[env=" << env_index << "] : "
                     << point_est_adj_gxe(j, phen_index)
                     << "  SE : " << SEjack_adj_gxe(se_row, 0) << endl;

                outfile << "h2_e[env=" << env_index << "] : "
                        << point_est_adj_gxe(j, phen_index)
                        << "  SE : " << SEjack_adj_gxe(se_row, 0) << endl;
            }
        }

        cout << "*****" << endl;
        outfile << "*****" << endl;
    }

    // ================= WIDE TABLE (one row per phenotype) =================
    print_wide_summary_table(point_est_preserved,
                             point_se,
                             SEjack_adj_gxe,
                             rho_hat_flat,
                             rho_se_flat);
}


void print_wide_summary_table(
    const MatrixXdr &point_est_preserved, // raw Gamma & Sigma^2
    const MatrixXdr &point_se,            // SE for raw components
    const MatrixXdr &SEjack_adj_gxe,      // SE for h2_g / h2_e
    const std::vector<double> &rho_hat_flat,
    const std::vector<double> &rho_se_flat)
{
    if (phenocount <= 0) return;

    int T_Nbin;
    if (gen_by_env == false) {
        T_Nbin = gen_Nbin;
    } else if (hetero_noise == true) {
        T_Nbin = gen_Nbin + nongen_Nbin + Nenv;
    } else {
        T_Nbin = gen_Nbin + nongen_Nbin;
    }

    const int nb_annot   = gen_Nbin;
    const int nb_gxe_tot = nongen_Nbin;
    const int nb_nxe_tot = T_Nbin - (gen_Nbin + nongen_Nbin);

    const int off_gxe = nb_annot;
    const int off_nxe = nb_annot + nb_gxe_tot;

    const int stride_gxe = (Nenv > 0 ? nb_gxe_tot / Nenv : 0);
    const int stride_nxe = (Nenv > 0 ? nb_nxe_tot / Nenv : 0);

    auto gxe_idx = [&](int e, int a) {
        return off_gxe + e * stride_gxe + a;
    };
    auto nxe_idx = [&](int e, int b) {
        return off_nxe + e * stride_nxe + b;
    };

    // Box–Cox info
    bool using_boxcox = (use_boxcox && !boxcox_lambdas.empty());
    int  L            = using_boxcox ? static_cast<int>(boxcox_lambdas.size()) : 0;

    // rho flat indexing
    bool have_rho = (!rho_hat_flat.empty() && Nenv > 1 && gen_Nbin > 0);

    auto rho_index = [&](int phen, int bin, int e1, int e2) -> std::size_t {
        // (((phen * gen_Nbin) + bin) * Nenv + e1) * Nenv + e2
        return ((((std::size_t)phen * (std::size_t)gen_Nbin + (std::size_t)bin)
                 * (std::size_t)Nenv + (std::size_t)e1)
                 * (std::size_t)Nenv + (std::size_t)e2);
    };

    auto get_rho = [&](int phen, int bin, int e1, int e2) -> double {
        if (!have_rho) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::size_t idx = rho_index(phen, bin, e1, e2);
        if (idx >= rho_hat_flat.size()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return rho_hat_flat[idx];
    };

    auto get_rho_se = [&](int phen, int bin, int e1, int e2) -> double {
        if (!have_rho) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::size_t idx = rho_index(phen, bin, e1, e2);
        if (idx >= rho_se_flat.size()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return rho_se_flat[idx];
    };

    // -------- Build column names (similar to Python wide_df) --------
    std::vector<std::string> cols;
    cols.push_back("phenotype");
    if (using_boxcox) {
        cols.push_back("lambda");
    }

    // Gamma (raw)
    for (int b = 0; b < gen_Nbin; ++b) {
        cols.push_back("gamma_bin" + std::to_string(b));
        cols.push_back("gamma_bin" + std::to_string(b) + "_se");
    }

    if (Nenv > 0) {
        // Sigma^2_g (raw)
        for (int e = 0; e < Nenv; ++e) {
            for (int b = 0; b < gen_Nbin; ++b) {
                cols.push_back("sigma2_g_env" + std::to_string(e) +
                               "_bin" + std::to_string(b));
                cols.push_back("sigma2_g_env" + std::to_string(e) +
                               "_bin" + std::to_string(b) + "_se");
            }
        }

        // Sigma^2_e (raw)
        for (int e = 0; e < Nenv; ++e) {
            cols.push_back("sigma2_e_env" + std::to_string(e));
            cols.push_back("sigma2_e_env" + std::to_string(e) + "_se");
        }

        // rho (bin, env-pair)
        if (have_rho) {
            for (int b = 0; b < gen_Nbin; ++b) {
                for (int e1 = 0; e1 < Nenv; ++e1) {
                    for (int e2 = e1 + 1; e2 < Nenv; ++e2) {
                        std::string epair = std::to_string(e1) + "-" + std::to_string(e2);
                        cols.push_back("rho_bin" + std::to_string(b) +
                                       "_env" + epair);
                        cols.push_back("rho_bin" + std::to_string(b) +
                                       "_env" + epair + "_se");
                    }
                }
            }
        }

        // h2_g (env-normalised)
        for (int e = 0; e < Nenv; ++e) {
            for (int b = 0; b < gen_Nbin; ++b) {
                cols.push_back("h2_g_env" + std::to_string(e) +
                               "_bin" + std::to_string(b));
                cols.push_back("h2_g_env" + std::to_string(e) +
                               "_bin" + std::to_string(b) + "_se");
            }
        }

        // h2_e (env-normalised)
        for (int e = 0; e < Nenv; ++e) {
            cols.push_back("h2_e_env" + std::to_string(e));
            cols.push_back("h2_e_env" + std::to_string(e) + "_se");
        }

        // P-values for env0 vs env1, bin0 (if present)
        if (Nenv >= 2 && gen_Nbin >= 1) {
            cols.push_back("pval.rho");
            cols.push_back("pval.sigma.diff");
            cols.push_back("pval.h2.diff");
        }
    }

    // Header
    outfile << "WIDE_TABLE" << std::endl;
    for (std::size_t j = 0; j < cols.size(); ++j) {
        if (j > 0) outfile << "\t";
        outfile << cols[j];
    }
    outfile << "\n";

    // Rows: one per phenotype
    for (int phen_index = 0; phen_index < phenocount; ++phen_index) {
        std::vector<double> row_vals;
        row_vals.reserve(cols.size());

        // phenotype index
        row_vals.push_back(static_cast<double>(phen_index));

        // lambda (if Box–Cox used)
        if (using_boxcox) {
            double lam = std::numeric_limits<double>::quiet_NaN();
            if (L > 0) {
                int l_idx = phen_index % L; // assuming phenos ordered by lambda
                if (l_idx >= 0 && l_idx < L) {
                    lam = boxcox_lambdas[l_idx];
                }
            }
            row_vals.push_back(lam);
        }

        // Gamma (raw)
        for (int b = 0; b < gen_Nbin; ++b) {
            double gamma_val = point_est_preserved(b, phen_index);
            int    se_row    = phen_index * T_Nbin + b;
            double gamma_se  = point_se(se_row, 0);
            row_vals.push_back(gamma_val);
            row_vals.push_back(gamma_se);
        }

        if (Nenv > 0) {
            // Sigma^2_g (raw)
            for (int e = 0; e < Nenv; ++e) {
                for (int b = 0; b < gen_Nbin; ++b) {
                    int idx     = gxe_idx(e, b);
                    double val  = point_est_preserved(idx, phen_index);
                    int    se_row = phen_index * T_Nbin + idx;
                    double se   = point_se(se_row, 0);
                    row_vals.push_back(val);
                    row_vals.push_back(se);
                }
            }

            // Sigma^2_e (raw) — assume one NxE component per env if present
            for (int e = 0; e < Nenv; ++e) {
                if (stride_nxe <= 0) {
                    row_vals.push_back(std::numeric_limits<double>::quiet_NaN());
                    row_vals.push_back(std::numeric_limits<double>::quiet_NaN());
                    continue;
                }
                int idx     = nxe_idx(e, 0);
                double val  = point_est_preserved(idx, phen_index);
                int    se_row = phen_index * T_Nbin + idx;
                double se   = point_se(se_row, 0);
                row_vals.push_back(val);
                row_vals.push_back(se);
            }

            // rho
            if (have_rho) {
                for (int b = 0; b < gen_Nbin; ++b) {
                    for (int e1 = 0; e1 < Nenv; ++e1) {
                        for (int e2 = e1 + 1; e2 < Nenv; ++e2) {
                            double rho = get_rho(phen_index, b, e1, e2);
                            double se  = get_rho_se(phen_index, b, e1, e2);
                            row_vals.push_back(rho);
                            row_vals.push_back(se);
                        }
                    }
                }
            }

            // h2_g (env-normalised)
            for (int e = 0; e < Nenv; ++e) {
                for (int b = 0; b < gen_Nbin; ++b) {
                    int idx     = gxe_idx(e, b);
                    double val  = point_est_adj_gxe(idx, phen_index);
                    int    se_row = phen_index * T_Nbin + idx;
                    double se   = SEjack_adj_gxe(se_row, 0);
                    row_vals.push_back(val);
                    row_vals.push_back(se);
                }
            }

            // h2_e (env-normalised)
            for (int e = 0; e < Nenv; ++e) {
                if (stride_nxe <= 0) {
                    row_vals.push_back(std::numeric_limits<double>::quiet_NaN());
                    row_vals.push_back(std::numeric_limits<double>::quiet_NaN());
                    continue;
                }
                int idx     = nxe_idx(e, 0);
                double val  = point_est_adj_gxe(idx, phen_index);
                int    se_row = phen_index * T_Nbin + idx;
                double se   = SEjack_adj_gxe(se_row, 0);
                row_vals.push_back(val);
                row_vals.push_back(se);
            }

            // p-values for env0 vs env1, bin0
            if (Nenv >= 2 && gen_Nbin >= 1) {
                double p_rho        = std::numeric_limits<double>::quiet_NaN();
                double p_sigma_diff = std::numeric_limits<double>::quiet_NaN();
                double p_h2_diff    = std::numeric_limits<double>::quiet_NaN();

                // pval.rho: test rho = 1 using rho_bin0_env0-1
                if (have_rho) {
                    double rho    = get_rho(phen_index, 0, 0, 1);
                    double rho_se = get_rho_se(phen_index, 0, 0, 1);
                    if (std::isfinite(rho) && rho >= 0.0 && rho_se > 0.0) {
                        double z = std::fabs(1.0 - rho) / rho_se;
                        p_rho = 2.0 * normal_sf(z);
                    }
                }

                // pval.sigma.diff: sigma2_g_env0_bin0 vs sigma2_g_env1_bin0 (raw)
                if (stride_gxe > 0) {
                    int idx0     = gxe_idx(0, 0);
                    int idx1     = gxe_idx(1, 0);
                    double s0    = point_est_preserved(idx0, phen_index);
                    double s1    = point_est_preserved(idx1, phen_index);
                    int    se_row0 = phen_index * T_Nbin + idx0;
                    int    se_row1 = phen_index * T_Nbin + idx1;
                    double se0   = point_se(se_row0, 0);
                    double se1   = point_se(se_row1, 0);

                    if (s0 > 0.0 && s1 > 0.0 && se0 > 0.0 && se1 > 0.0) {
                        double diff    = std::fabs(s0 - s1);
                        double se_diff = std::sqrt(se0 * se0 + se1 * se1);
                        double z       = diff / se_diff;
                        p_sigma_diff   = 2.0 * normal_sf(z);
                    }
                }

                // pval.h2.diff: h2_g_env0_bin0 vs h2_g_env1_bin0
                if (stride_gxe > 0) {
                    int idx0     = gxe_idx(0, 0);
                    int idx1     = gxe_idx(1, 0);
                    double h0    = point_est_adj_gxe(idx0, phen_index);
                    double h1    = point_est_adj_gxe(idx1, phen_index);
                    int    se_row0 = phen_index * T_Nbin + idx0;
                    int    se_row1 = phen_index * T_Nbin + idx1;
                    double se0   = SEjack_adj_gxe(se_row0, 0);
                    double se1   = SEjack_adj_gxe(se_row1, 0);

                    if (h0 > 0.0 && h1 > 0.0 && se0 > 0.0 && se1 > 0.0) {
                        double diff    = std::fabs(h0 - h1);
                        double se_diff = std::sqrt(se0 * se0 + se1 * se1);
                        double z       = diff / se_diff;
                        p_h2_diff      = 2.0 * normal_sf(z);
                    }
                }

                row_vals.push_back(p_rho);
                row_vals.push_back(p_sigma_diff);
                row_vals.push_back(p_h2_diff);
            }
        }

        // Write row
        for (std::size_t j = 0; j < row_vals.size(); ++j) {
            if (j > 0) outfile << "\t";
            double v = row_vals[j];
            if (std::isfinite(v)) {
                outfile << v;
            } else {
                outfile << "NA";
            }
        }
        outfile << "\n";
    }

    outfile << "*****" << std::endl;
}



void print_trace () {
	string prefix = command_line_opts.OUTPUT_FILE_PATH;
	string trpath=prefix + ".tr";
	string mnpath=prefix + ".MN";
	trace_file.open(trpath.c_str(), std::ios_base::out);
	stringstream ss;
	for (int i=0; i<Nbin; i++){
		ss << "LD_SUM_" << i << ",";
	}
	ss << "NSNPS_JACKKNIFE";
	trace_file << ss.str() << endl;
	meta_file.open(mnpath.c_str(), std::ios_base::out);
	meta_file << "NSAMPLE,NSNPS,NBLKS,NBINS,K" << endl << Nindv << "," << Nsnp << "," << Njack << "," << Nbin << "," << Nz;
	meta_file.close();
	trace_file.close();
}

void print_input_parameters() {
	string outpath = command_line_opts.OUTPUT_FILE_PATH;
	outfile.open(outpath.c_str(), std::ios_base::out);
	
	outfile << "##################################" << endl;
	outfile << "#                                #" << endl;
	outfile << "#         REGENA (v1.0.0)        #" << endl;
	outfile << "#                                #" << endl;
	outfile << "##################################" << endl;
	outfile << endl;
	outfile << endl;

	outfile << "Active essential options: " << endl;
	if (command_line_opts.GENOTYPE_FILE_PATH != "")
		outfile << "\t - g (genotype) " << command_line_opts.GENOTYPE_FILE_PATH << endl;
	if (command_line_opts.Annot_PATH != "")
		outfile << "\t - a (annotation) " << command_line_opts.Annot_PATH << endl;
	if (command_line_opts.PHENOTYPE_FILE_PATH != "")
		outfile << "\t - p (phenotype) " << command_line_opts.PHENOTYPE_FILE_PATH << endl;
	if (command_line_opts.COVARIATE_FILE_PATH != "")
		outfile << "\t - c (covariates) " << command_line_opts.COVARIATE_FILE_PATH << endl;
	if (command_line_opts.OUTPUT_FILE_PATH != "")
		outfile << "\t - o (output) " << command_line_opts.OUTPUT_FILE_PATH << endl;
	if (command_line_opts.ENV_FILE_PATH != "")
		outfile << "\t - e (environment) " << command_line_opts.ENV_FILE_PATH << endl;
	if (command_line_opts.model != "")
		outfile << "\t - m (model) " << command_line_opts.model << endl;
	if (command_line_opts.num_of_vec > 0) 
		outfile << "\t - k (# random vectors) " << std::to_string(command_line_opts.num_of_vec) << endl;
	if (command_line_opts.jack_number > 0)
		outfile << "\t - jn (# jackknife blocks) " << std::to_string(command_line_opts.jack_number) << endl;
	if (command_line_opts.nthreads > 0)
		outfile << "\t - t (# threads) " << std::to_string(command_line_opts.nthreads) << endl;
	if (command_line_opts.seed != -1)
		outfile << "\t - s (seed) " << std::to_string(command_line_opts.seed) << endl;
	if (command_line_opts.exannot == true)
		outfile << "\t - eXa (paritioned GxE)" << endl;
	if (verbose >= 1) {
		outfile << "Other options: " << endl;
		outfile << "\t -- norm_proj_pheno (normalize pheno after projection on covariates) " << std::to_string(command_line_opts.normalize_proj_pheno) << endl;
		outfile << "\t -- cov_add_intercept (intercept term added to covariates) " << std::to_string(command_line_opts.cov_add_intercept) << endl;
		outfile << "\t - v (verbose) " << std::to_string(command_line_opts.verbose) << endl;
	}
	if (trace)
		outfile << "\t - tr (trace summaries for G only; if pheno is not specified, dummy phenotype is used) " << endl;

	outfile << endl;
	outfile << endl;
	if ( verbose >= 3 ) {
		cout << "use_double = " << use_double << endl;
	}
}

void init_params () {

	verbose = command_line_opts.verbose;
	debug = command_line_opts.debugmode ||  verbose>= 3 ;

	trace = command_line_opts.print_trace;
	srand((unsigned int) time(0));
	Nz = command_line_opts.num_of_vec;
	k = Nz;

	jack_scheme = command_line_opts.jack_scheme;
	Njack = command_line_opts.jack_number;
	jack_size = command_line_opts.jack_size;
	jack_size *= 1e6; 

	memeff = command_line_opts.memeff;
	opt1 = command_line_opts.opt1;
	opt2 = command_line_opts.opt2;
	mem_Nsnp = command_line_opts.mem_Nsnp;
	use_mailman = command_line_opts.fast_mode;

	seed = command_line_opts.seed;
	if (command_line_opts.exannot == true)
		Annot_x_E = true;
	nthreads = command_line_opts.nthreads;
	
	cov_add_intercept = command_line_opts.cov_add_intercept;
	use_ysum = command_line_opts.use_ysum;
	keep_xsum = command_line_opts.keep_xsum;
}

template <typename Func>
void dummy_pheno(int Nind, Func& func){
	// fill in dummy phenotype
	cout << "Filling in dummy" << endl;
	mask.resize(Nindv, 1);
	pheno.resize(Nind, 1);
	for (int i=0; i < Nind; i++){
			pheno(i,0) = func();
			mask(i,0) = 1;

	}
}

int main(int argc, char const *argv[]){
    struct timeval now;
    gettimeofday(&now, NULL);
    long starttime = now.tv_sec * UMILLION + now.tv_usec;

	parse_args (argc,argv);
	init_params ();

	read_auxillary_files ();

	if (gen_by_env == true) {    
		///mask out indv with missingness from Enviro
		for (int i = 0; i < phenocount; i++)
				Enviro=Enviro.array().colwise()*mask.col(i).array();  
	}
	//define random vector z's
	all_zb = MatrixXdr::Random (Nindv, Nz);

	if (seed == -1) {
		std::random_device rd;
		seedr.seed(rd());
	} else {
		seedr.seed(seed);
	}

	std::normal_distribution<> dist(0,1);
	auto z_vec = std::bind(dist, seedr);

	for (int i = 0 ; i < Nz ; i++)
		for(int j = 0 ; j < Nindv ; j++)
			all_zb(j,i) = z_vec();
	
	// fill in dummy (for trace summaries)
	if (use_dummy)
		dummy_pheno(Nindv, z_vec);

	regress_covariates ();	

	for (int i = 0 ; i < Nz ; i++)
		for(int j = 0 ; j < Nindv ; j++)
			all_zb(j,i) = all_zb(j,i) * mask(j,0);

	if(both_side_cov == true){
		all_Uzb.resize(Nindv , Nz);
		for (int j = 0 ; j < Nz ; j++){
			MatrixXdr w1 = covariate.transpose() * all_zb.col(j);
			MatrixXdr w2 = Q * w1;
			MatrixXdr w3 = covariate * w2;
			all_Uzb.col(j) = w3;
		}
	}

	if(Annot_x_E == true){
		nongen_Nbin = Nenv * Nbin;
		for (int i = 0 ; i < Nenv ; i++)
			for (int j = 0 ; j < Nbin ; j++)
				len.push_back(len[j]);
	}else{
		nongen_Nbin = Nenv;
		for (int i = 0 ; i < Nenv ; i++)
			len.push_back(Nsnp);
	}


	// XXz : Nindv X (Nbins * Nz * 2): 
	// First half of columns has the computation from the current jackknife block.
	// Last half of columns has the computation from the whole genome.
	if(hetero_noise == true) {
		T_Nbin = Nbin + nongen_Nbin + Nenv;
	} else {
		T_Nbin = Nbin + nongen_Nbin;
	}

	if (memeff) { 
		XXz = MatrixXdr::Zero(Nindv, T_Nbin * Nz * 2);

		if(both_side_cov == true){
			UXXz = MatrixXdr::Zero(Nindv, T_Nbin * Nz * 2);
			XXUz = MatrixXdr::Zero(Nindv, T_Nbin * Nz * 2);
		}
		yXXy = MatrixXdr::Zero(phenocount*T_Nbin,2);

	} else {
		XXz=MatrixXdr::Zero(Nindv, T_Nbin * (Njack+1) * Nz);

		if(both_side_cov==true){
			UXXz=MatrixXdr::Zero(Nindv, T_Nbin * (Njack+1) * Nz);
			XXUz=MatrixXdr::Zero(Nindv, T_Nbin * (Njack+1) * Nz);
		}
		yXXy=MatrixXdr::Zero(phenocount*T_Nbin, Njack + 1);
	}

	if (!use_ysum)
		jack_yXXy = MatrixXdr::Zero(phenocount*T_Nbin, Njack);

	if(use_mailman == true) 
		allgen_mail.resize(Nbin);
	else
		allgen.resize(Nbin);

	int bin_index = 0;
	///// code for handling overlapping annotations
	string geno_name = command_line_opts.GENOTYPE_FILE_PATH;
	std::stringstream f3;
	f3 << geno_name << ".bed";
	string name = f3.str();
	ifstream ifs (name.c_str(), ios::in|ios::binary);
	read_header = true;
	global_snp_index=-1;

	if (!ifs.is_open()){
		cerr << "Error reading file "<< name  <<endl;
		exit(1);
	}

	cout << endl;
	cout << "Reading genotypes ..." << endl;

	print_input_parameters ();
	
	//CHANGE(03/05): add trace summary files. input to -o is now just the prefix (all output file endings are fixed to .log)
	if (trace){
		print_trace ();
	}

	// if (memeff) {
	// 	// This is where most of the computation happens
	// 	if (opt2){
	// 		genotype_stream_pass_mem_efficient (name);
	// 	} else {
	// 		genotype_stream_pass (name, 1);
	// 	}
	// 	genotype_stream_pass (name, 2);
	// } else {
	genotype_stream_single_pass (name);
	// }

	print_results ();

    gettimeofday(&now, NULL);
    long endtime = now.tv_sec * UMILLION + now.tv_usec;
    double elapsed = endtime - starttime;
    elapsed /= 1.e6;
    cout << "REGENA ran successfully. Time elapsed = " << elapsed << " seconds " << endl;
	
    outfile.close();
	return 0;
}
