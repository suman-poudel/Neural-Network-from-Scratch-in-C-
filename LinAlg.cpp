#include <iostream>
#include <vector>
#include <stdexcept>
#include "LinAlg.hpp"
#include <cmath>
#include <numeric>

using namespace std;

// elementwise multiplication of two matrices
vector<vector<double>> LinAlg::element_mat_mul(const vector<vector<double>>& A,
    const vector<vector<double>>& B) {
        int rows = A.size();
        int cols = A[0].size();

        vector<vector<double>> result (rows, vector<double> (cols));
        vector<vector<double>> sum_result (rows, vector<double> (1));

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
               result[i][j] = A[i][j] * B[i][j];
            }
          }

          return result;
    }


// Function to multiply two matrices
vector<vector<double>> LinAlg::mat_mul(const vector<vector<double>>& A,
                                        const vector<vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        cerr << "Matrix dimensions invalid for multiplication!" << endl;
        cerr << "A: " << A.size() << "x" << (A.empty() ? 0 : A[0].size()) << endl;
        cerr << "B: " << B.size() << "x" << (B.empty() ? 0 : B[0].size()) << endl;
        return {};
    }

    // Check if multiplication is possible
    if (colsA != rowsB) {
        throw invalid_argument("Matrix dimensions are incompatible for multiplication.");
    }

    // Result matrix initialized with 0s
    vector<vector<double>> result(rowsA, vector<double>(colsB, 0.0));

    // Multiply matrices
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}




vector<vector<double>> LinAlg::transpose_mat(const vector<vector<double>>& A) {
     if (A.empty() || A[0].empty()) return {}; // Handle empty matrix case
    int rowsA = A.size();
    int colsA = A[0].size();

    vector<vector<double>> transposed_mat(colsA, std::vector<double>(rowsA, 0));

    for (int i = 0; i < rowsA; ++i) {
      for (int j = 0; j < colsA; ++j) {
         transposed_mat[j][i] = A[i][j];
      }
    }

    return transposed_mat;
}




void LinAlg::show_matrix (const vector<vector<double>>& A) {
    int rowsA = A.size();
    int colsA = A[0].size();

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            cout << A[i][j] << " ";
        }
         cout << endl;
      }
}




vector<vector<double>> LinAlg::sum_mat(const vector<vector<double>>& A,
    const vector<vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != colsB && rowsA != rowsB) {
        throw invalid_argument("Matrix dimensions are incompatible for addition");
    }

    vector<vector<double>> result(rowsA, vector<double>(colsB, 0.0));

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {

            result[i][j] = A[i][j] + B[i][j];
    }

}

    return result;

}








vector<vector<double>> LinAlg::broadcast(const vector<vector<double>>& A,
    const vector<vector<double>>& B) {
    if (A.empty() || A[0].empty()) return {}; // Handle empty matrix case
    if (B.empty() || B[0].empty()) return {}; // Handle empty matrix case

    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    // Case 1: Broadcast 1-row matrix across rows
    if (rowsB == 1 && colsB == colsA) {
        vector<vector<double>> brod_B(rowsA, vector<double>(colsB));
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                brod_B[i][j] = B[0][j];
            }
        }
        return brod_B;
    }

    // Case 2: Broadcast 1-column matrix across columns
    if (colsB == 1 && rowsB == rowsA) {
        vector<vector<double>> brod_B(rowsB, vector<double>(colsA));
        for (int i = 0; i < rowsB; i++) {
            for (int j = 0; j < colsA; j++) {
                brod_B[i][j] = B[i][0];
            }
        }
        return brod_B;
    }

    // 🆕 Case 3: Broadcast scalar (1x1 matrix)
    if (rowsB == 1 && colsB == 1) {
        vector<vector<double>> brod_B(rowsA, vector<double>(colsA, B[0][0]));
        return brod_B;
    }

    if (rowsA == rowsB && colsA == colsB) {
        return B; // No broadcasting needed, return B as is
    }

    // Not broadcastable
    throw invalid_argument("Matrix dimensions are incompatible for broadcasting");
}







vector<vector<double>> LinAlg::square_loss(const vector<vector<double>>& A,
                                        const vector<vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    // Check if multiplication is possible
    if (colsA != colsB) {
        throw invalid_argument("Matrix dimensions are incompatible for multiplication.");
    }

    // Result matrix initialized with 0s
    vector<vector<double>> result(rowsA, vector<double>(colsA, 0.0));

    // Multiply matrices
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
                result[i][j] += pow(A[i][j] - B[i][j], 2);
            }
        }

    return result;
}



vector<vector<double>> LinAlg::sum_all_element(const vector<vector<double>>& A) {

    vector<vector<double>> sum (1, vector<double>(1, 0.0)); // Initialize sum to 0
    double sumation = 0;
    for (int i = 0; i < A.size(); ++i)
    {
        for (int j = 0; j < A[0].size(); ++j) {
                sumation += A[i][j];
            }
        }
    sum[0][0] = sumation;

    return sum;
}


vector<vector<double>> LinAlg::sum_row_element(const vector<vector<double>>& A) {

    vector<vector<double>> sum (1, vector<double>(A[0].size(), 0.0)); // Initialize sum to 0
    double sumation = 0;
    for (int j = 0; j < A[0].size(); ++j)
    {
        for (int i = 0; i < A.size(); ++i) {
                sumation += A[i][j];
            }
            sum[0][j] = sumation;
        }
    

    return sum;
}
