#include "auxillary.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

// Read environmental variable file
// Inputs: Number of individuals, filename
// Return number of environments
// Read environmental variable file
// Inputs: number of individuals, filename
// Returns: number of environments
int read_env (int Nind, std::string filename) {
    ifstream ifs(filename.c_str(), ios::in);
    if (!ifs.is_open()){
        cerr << "ERROR: Could not open environmental variable file: "
             << filename << endl;
        exit(1);
    }

    std::string line;
    std::istringstream in;
    std::getline(ifs, line);
    in.clear();
    in.str(line);

    string b;
    int xNenv = 0;
    // Count env columns (everything except FID/IID)
    while (in >> b) {
        if (b != "FID" && b != "IID") {
            xNenv++;
        }
    }

    vector<double> cov_sum(xNenv, 0.0);

    if (gen_by_env == true) {
        Enviro = MatrixXdr::Zero(Nind, xNenv);
        cout << "Reading " << xNenv << " environmental variables..." << endl;
    }

    // Are phenotypes/mask already initialized?
    bool have_mask =
        (phenocount > 0) &&
        (mask.rows() == Nind) &&
        (mask.cols() == phenocount);

    int j = 0;
    while (std::getline(ifs, line)) {
        if (j >= Nind) {
            cerr << "ERROR: More rows in env file than Nind = " << Nind << endl;
            exit(1);
        }

        in.clear();
        in.str(line);

        string temp;
        in >> temp; // FID
        in >> temp; // IID

        bool row_has_missing_env = false;

        for (int k = 0; k < xNenv; k++) {
            if (!(in >> temp)) {
                cerr << "ERROR: Not enough columns in env file at individual "
                     << j << endl;
                exit(1);
            }

            if (temp == "NA") {
                row_has_missing_env = true;
                continue;
            }

            double cur = atof(temp.c_str());
            if (cur == -9) {
                row_has_missing_env = true;
                continue;
            }

            if (gen_by_env == true) {
                cov_sum[k] += cur;
                Enviro(j, k) = cur;
            }
        }

        // Apply mask if we know about phenotypes
        if (have_mask && row_has_missing_env) {
            // Drop this individual for all phenotypes
            for (int l = 0; l < phenocount; l++) {
                mask(j, l) = 0;
            }
            // mask(j,0) is included in the loop above, so it’s consistent
        }

        j++;
    }

    // if (j != Nind) {
    //     cerr << "ERROR: Env file has " << j << " individuals, but Nind = "
    //          << Nind << endl;
    //     exit(1);
    // }

    return xNenv;
}

int read_env_bin (int Nind, std::string filename){
    std::ifstream ifs(filename.c_str(), std::ios::in);
    if (!ifs.is_open()){
        std::cerr << "Error reading file " << filename << std::endl;
        std::exit(1);
    }

    std::string line;
    std::istringstream in;

    // ---- Parse header, count env columns ----
    std::getline(ifs, line);
    in.str(line);
    std::string tok;
    int xNenv = 0;
    while (in >> tok){
        if (tok != "FID" && tok != "IID") xNenv++;
    }

    // Only support 1 col (E -> [E,1-E]) or 2 cols already [E, 1-E]
    if (xNenv != 1 && xNenv != 2){
        std::cerr << "read_env_bin error: expected 1 binary env column (to expand to [E,1-E]) "
                  << "or exactly 2 columns already [E,1-E]; got " << xNenv << "." << std::endl;
        std::exit(1);
    }

    const double eps = 1e-8;  // tolerance for binary checks
    auto is_binary = [&](double v)->bool {
        return (std::fabs(v) <= eps) || (std::fabs(v - 1.0) <= eps);
    };

    const bool expand_to_complement = (xNenv == 1);
    const int outNenv = 2; // we always end up with 2 columns: [E, 1-E]

    if (gen_by_env) {
        Enviro = MatrixXdr::Zero(Nind, outNenv);
        std::cout << "Reading " << xNenv << " env var(s)"
                  << (expand_to_complement ? " -> using 2 columns [E, 1-E]" : " (expecting [E, 1-E])")
                  << " ..." << std::endl;
    }

    int nonbin_count = 0;
    int noncomp_count = 0;
    int j = 0;

    while (std::getline(ifs, line)){
        in.clear();
        in.str(line);
        std::string tmp;
        // skip FID IID
        in >> tmp; in >> tmp;

        if (expand_to_complement){
            // Read exactly one env value
            if (!(in >> tmp)) { 
                for (int l = 0; l < phenocount; l++) {
                    mask(j, l) = 0;
                }
                j++; 
                continue; 
            }
            if (tmp == "NA" || tmp == "-9"){ 
                for (int l = 0; l < phenocount; l++) {
                    mask(j, l) = 0;
                } 
                j++; 
                continue; 
            }

            double e = std::atof(tmp.c_str());
            if (!is_binary(e)){
                nonbin_count++;
            }

            if (gen_by_env){
                // Store as-is (no clamping): we validated binary above
                Enviro(j, 0) = e;
                Enviro(j, 1) = 1.0 - e;
            }
        } else {
            // Expect two columns already [E, 1-E]
            std::string t0, t1;
            if (!(in >> t0)) { 
                for (int l = 0; l < phenocount; l++) {
                    mask(j, l) = 0;
                }  
                j++; 
                continue; 
            }
            if (t0 == "NA" || t0 == "-9") { 
                for (int l = 0; l < phenocount; l++) {
                    mask(j, l) = 0;
                }  
                j++; 
                continue; 
            }
            if (!(in >> t1)) { 
                for (int l = 0; l < phenocount; l++) {
                    mask(j, l) = 0;
                }  
                j++; 
                continue; 
            }
            if (t1 == "NA" || t1 == "-9") { 
                for (int l = 0; l < phenocount; l++) {
                    mask(j, l) = 0;
                }  
                j++; 
                continue; 
            }

            double e0 = std::atof(t0.c_str());
            double e1 = std::atof(t1.c_str());

            if (!is_binary(e0) || !is_binary(e1)){
                nonbin_count++;
            }
            if (std::fabs(e0 + e1 - 1.0) > eps){
                noncomp_count++;
            }

            if (gen_by_env){
                Enviro(j, 0) = e0;
                Enviro(j, 1) = e1;
            }
        }
        j++;
    }

    // Complain if not binary / not complementary
    if (nonbin_count > 0){
        std::cerr << "read_env_bin error: environment must be binary (0/1). "
                  << "Found " << nonbin_count << " non-binary row(s)." << std::endl;
        std::exit(1);
    }
    if (!expand_to_complement && noncomp_count > 0){
        std::cerr << "read_env_bin error: with 2 env columns, expected [E, 1-E] per row. "
                  << "Found " << noncomp_count << " non-complementary row(s)." << std::endl;
        std::exit(1);
    }

    return outNenv;
}

// Read covariate file. 
// Read environmental variables (and Enviro is initialized) before calling this function.
// Adds environmental variables to covariates if add_env_to_cov == True
// Adds intercept to covariates if cov_add_intercept == True
// Inputs: Number of individuals, filename
// Return total number of covariates (covariates specified in covariate file [ + environmental variables] [ + intercept])
int read_cov (int Nind, std::string filename) {
    ifstream ifs(filename.c_str(), ios::in);
    if (!ifs.is_open()){
        cerr << "ERROR: Error reading covariate file : "<< filename << endl;
        exit(1);
    }
    std::string line;
    std::istringstream in;
    int covIndex = 0;
    std::getline(ifs, line);
    in.str(line);
    string b;
    vector<vector<int> > missing;
    int covNum = 0;
    int Nenv = 0;
    if (add_env_to_cov == true)
        Nenv = Enviro.cols();

    // We will potentially fix redundancy only in this regime:
    bool allow_env_redundancy_fix = (gen_by_env && add_env_to_cov && cov_add_intercept && Nenv >= 1);

    vector<double> cov_sum;
    while (in >> b) {
        if (b != "FID" && b != "IID") {
            missing.push_back(vector<int>()); // push an empty row
            covNum++;
        }
    }

    if (covNum != 0)
        cov_sum.resize(covNum, 0.0);

    if (cov_add_intercept == true) {
        if (add_env_to_cov == true)
            covariate.resize(Nind, covNum + Nenv + 1);
        else
            covariate.resize(Nind, covNum + 1);
    } else {
        if (add_env_to_cov == true)
            covariate.resize(Nind, covNum + Nenv);
        else
            covariate.resize(Nind, covNum);
    }

    cout << "Reading in " << covNum << " covariates ..." << endl;

    int j = 0;
    while (std::getline(ifs, line)) {
        in.clear();
        in.str(line);
        string temp;
        in >> temp; // FID
        in >> temp; // IID
        for (int k = 0; k < covNum; k++) {
            in >> temp;
            if (temp == "NA") {
                missing[k].push_back(j);
                continue;
            }
            double cur = atof(temp.c_str());
            if (cur == -9) {
                missing[k].push_back(j);
                continue;
            } else {
                cov_sum[k] += cur;
                covariate(j, k) = cur;
            }
        }
        j++;
    }

    // compute cov mean and impute
    for (int a = 0; a < covNum; a++) {
        int missing_num = missing[a].size();
        cov_sum[a] = cov_sum[a] / (Nind - missing_num);

        for (int b = 0; b < missing_num; b++) {
            int index = missing[a][b];
            covariate(index, a) = cov_sum[a];
        }
    }

    if (gen_by_env == true) {
        if (verbose >= 1) {
            cout << "Shape of Env = " << Enviro.rows() << " " << Enviro.cols() << endl;
            cout << "Shape of Cov = " << covariate.rows() << " " << covariate.cols() << endl;
            cout << "Number of covariates (file) = " << covNum
                 << ", number of environments = " << Nenv << endl;
        }
        if (add_env_to_cov == true) {
            // append raw environment columns as covariates
            for (int i = 0; i < Nenv; i++) {
                covariate.col(covNum + i) = Enviro.col(i);
            }
        }
    } else {
        if (cov_add_intercept == true) {
            // adding col of all ones to covariates
            for (int i = 0; i < Nind; i++)
                covariate(i, covNum + Nenv) = 1.0;
            return covNum + 1;
        }
    }

    // add intercept at the end (after any env columns)
    if (cov_add_intercept == true) {
        for (int i = 0; i < Nind; i++)
            covariate(i, covNum + Nenv) = 1.0;
    }

    // ----------------- Redundancy detection & automatic fix -----------------
    int effective_Nenv = Nenv;
    bool dropped_env_col = false;

    // Total covariate columns *before* any dropping
    int Ncov_total;
    if (cov_add_intercept == true) {
        if (add_env_to_cov == true)
            Ncov_total = covNum + Nenv + 1;
        else
            Ncov_total = covNum + 1;
    } else {
        if (add_env_to_cov == true)
            Ncov_total = covNum + Nenv;
        else
            Ncov_total = covNum;
    }

    if (allow_env_redundancy_fix && Nenv >= 2) {
        // Check if Enviro is one-hot across columns (row-sum ~ 1)
        double max_diff = 0.0;
        for (int i = 0; i < Nind; ++i) {
            double s = 0.0;
            for (int e = 0; e < Nenv; ++e)
                s += Enviro(i, e);
            max_diff = std::max(max_diff, std::fabs(s - 1.0));
        }

        if (max_diff < 1e-6) {
            // We have one-hot env + intercept -> linear dependency: sum(env) == intercept.
            // Drop the last env column from covariate (index covNum + Nenv - 1).
            int drop_col = covNum + Nenv - 1;
            int new_cols = Ncov_total - 1;

            if (verbose >= 1) {
                cerr << "[WARN] Detected one-hot environment coding with intercept in read_cov().\n"
                     << "       Enviro has " << Nenv << " columns whose row-sums are all 1,\n"
                     << "       and add_env_to_cov == true, cov_add_intercept == true.\n"
                     << "       To avoid linear dependence, dropping env column index "
                     << drop_col << " from covariates (keeping intercept).\n";
            }

            MatrixXdr cov_new(Nind, new_cols);
            int out_c = 0;
            for (int c = 0; c < Ncov_total; ++c) {
                if (c == drop_col) continue;
                cov_new.col(out_c++) = covariate.col(c);
            }
            covariate = cov_new;
            Ncov_total = new_cols;
            effective_Nenv = Nenv - 1;
            dropped_env_col = true;
        }
    }

    // Second pass: constant-column check (after any dropping)
    for (int c = 0; c < Ncov_total-1; ++c) {
        double v0 = covariate(0, c);
        bool all_same = true;
        for (int i = 1; i < Nind; ++i) {
            if (covariate(i, c) != v0) {
                all_same = false;
                break;
            }
        }
        if (all_same) {
            cerr << "[WARN] Covariate column " << c
                 << " is constant (value = " << v0 << "). "
                 << "This column is redundant and may cause instability.\n";
        }
    }
    // ----------------- End redundancy logic -----------------

    // Return effective number of covariates
    if (cov_add_intercept == true) {
        if (add_env_to_cov == true)
            return covNum + effective_Nenv + 1;
        else
            return covNum + 1;
    } else {
        if (add_env_to_cov == true)
            return covNum + effective_Nenv;
        else
            return covNum;
    }
}


// Read pheno file
// Inputs: Number of individuals, filename
void read_pheno(int Nind, std::string filename){
	ifstream ifs(filename.c_str(), ios::in); 

	if (!ifs.is_open()){
		cerr << "ERROR: Error reading phenotype file : "<< filename <<endl;
		exit(1);
	}

	std::string line;
	std::istringstream in;	
	phenocount = 0; 
	//read header
	std::getline(ifs,line); 
	in.str(line); 
	string b; 
	while(in>>b)
	{
		if(b!="FID" && b !="IID")
			phenocount++; 
	}

	// if (phenocount > 1) { 
	// 	cerr << "ERROR: The phenotype file can only have a single phenotype" << endl;
	// 	exit (1);
	// }

	pheno.resize(Nind, phenocount);
	new_pheno.resize(Nind, phenocount);
	mask.resize(Nind, phenocount);
	int i = 0;	
	while(std::getline(ifs, line)){
		in.clear(); 
		in.str(line); 
		string temp;
		//fid,iid
		//todo: fid iid mapping; 
		//todo: handle missing phenotype
		in>>temp; in>>temp; 
		for(int j = 0; j < phenocount ; j++) {
			in>>temp;
			double cur = atof(temp.c_str());
			if(temp=="NA" || cur==-9){
				pheno(i,j) = 0;
				mask(i,j) = 0;
				mask(i,0) = 0; // use the overlapping individuals
			}
			else{
				pheno(i,j) = atof(temp.c_str());
				mask(i,j) = 1;
			}
		}
		i++;
	}
}

//
// Count number of individuals in .pheno file
// filename: name of .pheno file
int count_pheno(std::string filename){
	ifstream ifs(filename.c_str(), ios::in);
	if (!ifs.is_open()){
		cerr << "ERROR: Error reading phenotype file : "<< filename <<endl;
		exit(1);
	}

	std::string line;
	int i = 0;
	while(std::getline(ifs, line)){
		i++;
	}
	int Nindv = i - 1;
	return Nindv;
}

void read_annot (string filename) {
	ifstream inp(filename.c_str());
	if (!inp.is_open()){
		cout <<"WARNING: no annotation file provided. All SNPs will be assigned to a single annotation" << endl;	
		Nbin = 1;
		vector<bool> snp_annot;
		if(Annot_x_E == false)
			snp_annot.resize(Nbin + Nenv, 1);
		else
			snp_annot.resize(Nbin + (Nenv * Nbin), 1);

		for (int i = 0 ; i < Nsnp ; i++) 
			annot_bool.push_back(snp_annot);

		len.resize(1, Nsnp);

	} else {
		string line;
		int linenum = 0 ;
		int num_parti;
		stringstream check1(line);
		string intermediate;
		vector <string> tokens;
		vector<bool> snp_annot;

		while(std::getline (inp, line)){
			char c = line[0];	
			if (c=='#')
				continue;
			istringstream ss (line);
			if (line.empty())
				continue;

			stringstream check1(line);
			string intermediate;
			vector <string> tokens;
			// Tokenizing w.r.t. space ' ' 
			while(getline(check1, intermediate, ' '))
			{
				tokens.push_back(intermediate);
			}
			if(linenum == 0){
				num_parti = tokens.size();
				Nbin = num_parti;
				if(Annot_x_E == false)
					snp_annot.resize(Nbin + Nenv, 0);
				else
					snp_annot.resize(Nbin + (Nenv * Nbin), 0);

				len.resize(num_parti, 0);
			}

			if (tokens.size () != Nbin) {
				cerr << "ERROR: Error reading annotation file : "<< filename << " at line number : " << linenum <<endl;
				exit(1);
			}

			int index_annot = 0;
			for(int i = 0; i < tokens.size(); i++){
				snp_annot[i] = 0;
				if (tokens[i] == "1"){
					len[i] ++;
					snp_annot[i] = 1;
				}
			}
			if(Annot_x_E == false){
				for(int i = 0 ; i < Nenv ; i++)
					snp_annot[Nbin + i] = 1;   /// need to modify for excluding snps
			} else {
				for(int i = 0 ; i < Nenv ; i++)
					for(int j = 0 ; j < Nbin ; j++)
						snp_annot[Nbin + (i * Nbin) + j] = snp_annot[j];
			}
			annot_bool.push_back(snp_annot);
			linenum++;
		}
		if(Nsnp != linenum){
			cerr << "Number of rows in bim file and annotation file does not match\n" << endl;
			exit (1);
		}
	}

	//cout << "Total number of SNPs : " << Nsnp << endl;
	cout << "Number of annotations in annotation file = " << Nbin << endl;
	int selected_snps = 0;
	for (int i = 0 ; i < Nbin ; i++){
		cout << "Number of SNPs in annotation " << i << " = " << len[i] <<endl;
		selected_snps += len[i];
	}

	cout << "Number of SNPs selected according to annotation file = " <<selected_snps << endl;
	Nsnp_annot = selected_snps;

	int Total_Nbin;
	if(Annot_x_E == false){
		jack_bin.resize(Njack, vector<int>(Nbin + Nenv + Nenv,0));
		if (memeff)
			read_bin.resize(Nreadblocks, vector<int>(Nbin + Nenv + Nenv,0));
		Total_Nbin = Nbin + Nenv;
	} else {
		jack_bin.resize(Njack, vector<int>(Nbin + (Nenv * Nbin) + Nenv,0));
		if (memeff)
			read_bin.resize(Nreadblocks, vector<int>(Nbin + (Nenv * Nbin) + Nenv,0));
		Total_Nbin = Nbin + (Nenv * Nbin);
	}

	for (int i = 0 ; i < Nsnp ; i++)
		for(int j = 0 ; j < Total_Nbin ; j++)
			if (annot_bool[i][j]==1){
				int temp = snp_to_jackblock[i];
				jack_bin[temp][j]++;

				if (memeff)  {
					temp = snp_to_read_block[i];
					read_bin[temp][j]++;
				}
			}

	vector<int> bin_totals(Total_Nbin, 0);
	for (int b = 0; b < Total_Nbin; ++b) {
		long tot = 0;
		for (int jb = 0; jb < Njack; ++jb) tot += jack_bin[jb][b];
		bin_totals[b] = (int)tot;
	}
		// Error if any jackknife block contains ALL SNPs of a bin
	for (int b = 0; b < Total_Nbin; ++b) {
		const int total = bin_totals[b];
		if (total <= 0) continue; // skip empty bins
		for (int jb = 0; jb < Njack; ++jb) {
			if (jack_bin[jb][b] == total) {
				cerr << "ERROR: Jackknife block " << jb
					 << " contains ALL " << total
					 << " SNPs for bin " << b << ".\n"
					 << "This makes leave-one-out undefined and invalidates SEs.\n"
					 << "Consider shuffling SNP order, changing Njack, or a different jackknife scheme.\n";
				exit(1);
			}
		}
	}
}

void read_annot_1col (string filename){
	ifstream ifs(filename.c_str(), ios::in);

	std::string line;
	std::istringstream in;

	len.resize(Nbin,0);
	step_size = Nsnp / Njack;
	step_size_rem = Nsnp % Njack;
	cout << "Number of SNPs per block : " << step_size << endl;
	jack_bin.resize(Njack, vector<int>(Nbin,0));
	int i = 0;
	while(std::getline(ifs, line)){
		in.clear();
		in.str(line);
		string temp;

		in>>temp;		 
		int cur = atoi(temp.c_str());
		SNP_annot.push_back(cur);
		len[cur - 1]++;

		int jack_val = i / step_size;
		if (jack_val == Njack)
			jack_val--;
		jack_bin[jack_val][SNP_annot[i]-1]++;
		i++;
	}

	if(Nsnp != i){
		cerr << "Number of rows in bim file and annotation file does not match" << endl;
		exit (1);
	}
	cout << "Total number of SNPs : " << Nsnp << endl;
	for (int i = 0 ; i < Nbin ; i++){
		cout << len[i]<<" SNPs in " << i<<"-th bin" << endl;
		Nsnp_annot += len[i];
	}
}
