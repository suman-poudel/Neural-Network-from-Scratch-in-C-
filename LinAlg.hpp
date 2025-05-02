#ifndef _DATA_H
#define _DATA_H


#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

class LinAlg {
    public:
        static vector<vector<double>> mat_mul(const vector<vector<double>>& , const vector<vector<double>>& );
        static vector<vector<double>> element_mat_mul(const vector<vector<double>>& ,
            const vector<vector<double>>& );
        static vector<vector<double>> transpose_mat(const vector<vector<double>>& );
        static void show_matrix (const vector<vector<double>>& );
        static vector<vector<double>> sum_mat(const vector<vector<double>>& , const vector<vector<double>>& );
        static vector<vector<double>> broadcast(const vector<vector<double>>& , const vector<vector<double>>& );
        static vector<vector<double>> square_loss(const vector<vector<double>>& A, const vector<vector<double>>& B);
        static vector<vector<double>> sum_all_element(const vector<vector<double>> &);
        static vector<vector<double>> sum_row_element(const vector<vector<double>> &);
};

#endif
