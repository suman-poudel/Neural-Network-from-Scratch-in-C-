#include<iostream>
#include<vector>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include "LinAlg.hpp"

using namespace std;

// Datasets
pair<vector<vector<double>>, vector<vector<int>>> spiral_data (int points, int classes) {

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> noise(0, 0.2); // mean=0, std=0.2


    vector<vector<double>> X(points * classes, vector<double> (2, 0));
    vector<vector<int>> y(points * classes, vector<int> (1, 0));

    for (int class_number = 0; class_number < classes; ++class_number) {
        for (int i = 0; i < points; i++) {
            int ix = class_number * points + i;
            double r = static_cast<double>(i) / (points - 1);  // linspace from 0 to 1
            double t = class_number * 4 + (4.0 * i / (points - 1)) + noise(gen);  // angle with noise

            X[ix][0] = r * sin(t * 2.5);
            X[ix][1] = r * cos(t * 2.5);
            y[ix][0] = class_number;
        }
    }

    return {X, y};
}

// Class for definning the neural network layer
class Layer_Dense {
    private:
        std::mt19937 gen;
        std::uniform_real_distribution<> dis;
    public:
        vector<vector<double>> weights;
        vector<vector<double>> biases;
        vector<vector<double>> dL_dW;
        vector<vector<double>> dL_db;
        vector <vector<double>> dL_dinputs;
        vector<vector<double>> inputs;
        Layer_Dense (int n_inputs, int n_neurons)
        :gen(std::random_device{}()), dis(-1.0, 1.0) {
            
            weights.resize(n_inputs, vector<double> (n_neurons));
            biases.resize(1, vector<double> (n_neurons, 0));

            double stddev = sqrt(2.0/n_inputs);
            normal_distribution<> dis(0, stddev);

            for (int i = 0; i < n_inputs; ++i) {
                for (int j = 0; j < n_neurons; ++j) {
                    weights[i][j] = dis(gen);
                }
            }
        }

        vector<vector<double>> forward (vector<vector<double>> inputs1) {
            this -> inputs = inputs1;
            vector <vector<double>> mult_out = LinAlg::mat_mul(inputs1, weights);
            vector <vector<double>> brod_bias = LinAlg::broadcast(mult_out, biases);
            vector <vector<double>> layer_output = LinAlg::sum_mat(mult_out, brod_bias);

            return layer_output;
        }

        void backward (vector<vector<double>> dL_dZ) {
            
            dL_dW = LinAlg::mat_mul(LinAlg::transpose_mat(inputs), dL_dZ);
            // LinAlg::show_matrix(dL_dW);
            // cout << endl;
            // cout << "Layer_Dense Backward dL_dW: " << dL_dW.size() << "x" << dL_dW[0].size() << endl;

            dL_db = LinAlg::sum_row_element(dL_dZ);
            // cout << "Layer_Dense Backward dL_db: " << dL_db.size() << "x" << dL_db[0].size() << endl;

            // cout << "weights: " << weights.size() << "x" << weights[0].size() << endl;
            // cout << "inputs: " << dL_dZ.size() << "x" << dL_dZ[0].size() << endl;
            this -> dL_dinputs = LinAlg::mat_mul(dL_dZ, LinAlg::transpose_mat(weights));
            // LinAlg::show_matrix(dL_dinputs);
            // cout << endl;
            // cout << "Layer_Dense Backward dL_dinputs: " << dL_dinputs.size() << "x" << dL_dinputs[0].size() << endl;
        }

};

// ReLU Activation class

class Activation_ReLU {
    public:
        vector<vector<double>> layer;
        vector<vector<double>> forward (vector<vector<double>> layer_output) {
            layer = layer_output;
            int rows = layer_output.size();
            int cols = layer_output[0].size();

            for (int i=0; i < rows; i++) {
                for (int j=0; j < cols; j++) {
                    layer_output[i][j] =  max(0.0, layer_output[i][j]);
                }
            }

            return layer_output;
        }

        vector<vector<double>> backward (vector<vector<double>> dvalues) {
            vector<vector<double>> dinputs = dvalues;
            // cout << "dinputs: " << dinputs.size() << "x" << dinputs[0].size() << endl;
            // cout << "layer: " << layer.size() << "x" << layer[0].size() << endl;
            // LinAlg::show_matrix(layer);
            // cout << endl;

            for (int i=0; i < layer.size(); i++) {
                for (int j=0; j < layer[0].size(); j++) {
                    if (layer[i][j] <= 0) {
                        dinputs[i][j] = 0;
                    }
                    // cout << dinputs1[i][j] << " ";
                }
                // cout << endl;
            }

            // LinAlg::show_matrix(dinputs1);
            // cout << endl;
            

            return dinputs;
        }

        vector<vector<double>> relu_derivative(const vector<vector<double>>& input) {
        vector<vector<double>> output = input;
        for (size_t i = 0; i < input.size(); ++i) {
            for (size_t j = 0; j < input[i].size(); ++j) {
                output[i][j] = (input[i][j] > 0.0) ? 1.0 : 0.0;
            }
        }
    return output;
}
};


// Softmax Activation class
class Activation_Softmax {
    public:
        vector<vector<double>> normalize_rows(const vector<vector<double>>& data) {
            vector<vector<double>> normalized;
        
            for (const auto& row : data) {
                double sum = accumulate(row.begin(), row.end(), 0.0);
                vector<double> norm_row;
                for (double val : row) {
                    norm_row.push_back(val / sum);
                }
                normalized.push_back(norm_row);
            }
        
            return normalized;
        }


        vector<vector<double>> forward (vector<vector<double>> inputs) {
            vector<vector<double>> exp_values;
            double max_val = inputs[0][0];

            vector<double> row_max;

            for (const auto& row:inputs) {
                row_max.push_back(*max_element(row.begin(), row.end()));
            }

        for (int i = 0; i < inputs.size(); i++) {
            vector<double> rows;
            for (int j = 0; j < inputs[0].size(); j++) {
                rows.push_back(exp(inputs[i][j] - row_max[i]));
            }
            exp_values.push_back(rows);
        }
        return normalize_rows(exp_values);
        }
};


// Categorical loss function
class Loss_CategoricalCrossentopy {

    public:
        vector<vector<double>>  forward (const vector<vector<double>>&  y_pred, const vector<vector<int>>&  y_true) {
            int samples = y_pred[0].size();

            vector<vector<double>> y_pred_clipped(y_pred.size(), vector<double> (y_pred[0].size(), 0));
            const double epsilon = 1e-7;
            for (int i = 0; i < y_pred.size(); i++) {
                for (int j = 0; j < y_pred[0].size();  j++) {
                    y_pred_clipped[i][j] = min(max(y_pred[i][j], epsilon), 1.0 - epsilon);
                }
            }

            bool is_sparse = y_true[0].size() == 1;

            if (is_sparse) {
                vector<vector<double>> correct_confidences;
                vector<vector<double>> neg_log;

                    vector<double> rows;
                    for (int i = 0; i < y_pred.size(); i++) {
                        if (y_true[i][0] < 0 || y_true[i][0] >= y_pred[i].size()) {
                            cerr << "Error: Invalid label index at sample " << i << endl;
                            exit(1);
                            }
                        correct_confidences.push_back({y_pred[i][y_true[i][0]]});
                        neg_log.push_back({-log(y_pred[i][y_true[i][0]])});
                    }

                    return neg_log;

            } else {
                int rows = y_pred.size();
                int cols = y_pred[0].size();

                vector<vector<double>> result (rows, vector<double> (cols));
                vector<vector<double>> correct_confidences (rows, vector<double> (1));
                vector<vector<double>> neg_log (rows, vector<double> (1));

                for (int i = 0; i < rows; i++) {
                double temp_sum = 0;
                    for (int j = 0; j < cols; j++) {
                    result[i][j] = y_pred[i][j] * y_true[i][j];
                    temp_sum += result[i][j];
                }
                correct_confidences[i][0] = temp_sum;
                neg_log[i][0] = -log(temp_sum);
                // cout << sum_result[i][0] << endl;
            }

            return neg_log;
            }

        }

        vector<vector<double>>  backward (const vector<vector<double>>&  y_pred, const vector<vector<int>>&  y_true) {
            int samples = y_pred[0].size();
            int classes = y_pred.size();

            vector<vector<double>> dvalues (classes, vector<double> (samples, 0));

            for (int i = 0; i < classes; i++) {
                for (int j = 0; j < samples; j++) {
                    dvalues[i][j] = - y_true[i][j]/(y_pred[i][j] * samples);
                }
            }

            return dvalues;
        }

        double calculate (vector<vector<double>> output,  vector<vector<int>> y) {
            vector<vector<double>> sample_losses = forward(output, y);
            double data_loss = 0;
            for (int i = 0; i < sample_losses.size(); i++) {
                for (int j = 0; j < sample_losses[0].size(); j++) {
                    data_loss += sample_losses[i][j];
                }
            }
                return data_loss/sample_losses.size();
    }
};


class Activation_Softmax_Loss_CategoricalCrossentropy {
    public:
        vector<vector<double>> softmax_output;
        Activation_Softmax softmax;
        Loss_CategoricalCrossentopy loss_function;

        double forward (const vector<vector<double>>& inputs, const vector<vector<int>>& y_true) {
            this -> softmax_output = softmax.forward(inputs);
            double loss = loss_function.calculate(softmax_output, y_true);
            return loss;
        }

        vector<vector<double>> backward(const vector<vector<double>>& y_pred, const vector<vector<int>>& y_true) {
    vector<vector<double>> dvalues = y_pred;
    int batch_size = y_true.size();
    
    if (y_true[0].size() == 1) {  // Sparse labels (Nx1)
        for (int i = 0; i < batch_size; i++) {
            int true_class = y_true[i][0];
            dvalues[i][true_class] -= 1.0;
        }
    } 
    else {  // One-hot encoded labels (NxC)
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < y_true[0].size(); j++) {
                if (y_true[i][j] == 1) {
                    dvalues[i][j] -= 1.0;
                }
            }
        }
    }
    
    // Normalize by batch size
    for (auto& row : dvalues) {
        for (auto& val : row) {
            val /= batch_size;
        }
    }
    
    return dvalues;
}
    };


class Optimizer_SGD {
    public:
        double learning_rate = 0.01;
        double current_learning_rate = 0.01;
        int iterations = 0;
        double decay = 0.0;
        double momentum = 0.0;
        vector<vector<double>> weights;
        vector<vector<double>> biases;

        Optimizer_SGD (double learning_rate, double decay, double momentum) {
            this -> learning_rate = learning_rate;
            this -> current_learning_rate = learning_rate;
            this -> decay = decay;
            this -> momentum = momentum;
            this -> iterations = 0;
        }

        double pre_update_params () {
            if (decay) {
                current_learning_rate = learning_rate / (1 + decay * iterations);
            }
            return current_learning_rate;
        }


        vector<vector<double>> update_weight (vector<vector<double>>& weights, vector<vector<double>>& dL_dW) {

            if (momentum) {
                //  implement momentum update;
                ;
            } else {
            for (int i = 0; i < weights.size(); i++) {
                for (int j = 0; j < weights[0].size(); j++) {
                    weights[i][j] += -current_learning_rate * dL_dW[i][j];
                }
            }
            return weights;
        }
        
        }

    void post_update_params () {
        iterations += 1;
    }

};





auto check_gradients = [](const vector<vector<double>>& grad) {
    for (const auto& row : grad) {
        for (double val : row) {
            if (isnan(val) || isinf(val)) {
                cerr << "Invalid gradient value detected!" << endl;
                exit(1);
            }
        }
    }
};








int main () {

    srand(123);

    using namespace std::chrono;

    auto start = high_resolution_clock::now();  // Start timer

    // Code to time
    for (int i = 0; i < 1e7; ++i) {
        volatile int x = i * i;  // Prevent compiler optimization
    }


    auto result = spiral_data(100, 3);
    vector <vector<double>>  X = result.first;
    vector <vector<int>>  y = result.second;

    // for (int i = 0; i < y.size(); i++) {
    //     for (int j = 0; j < y[0].size(); j++) {
    //         cout << "y_value" << y[i][j] << " ";
    //     }

    //     cout << endl;
    // }

    Layer_Dense dense1(X[0].size(), 3);
    Activation_ReLU ReLU;
    Layer_Dense dense2(3, 3);
    Activation_Softmax_Loss_CategoricalCrossentropy Loss_Softmax;
    double learning_rate_init = 1;
    double learning_rate = learning_rate_init;
    Optimizer_SGD optimizer(1, 0.001, 0.0);




    // ***************  Loop Starts  ************************
    for (int k = 0; k < 1000; k++) {
        vector<vector<double>> layer_out1 = dense1.forward(X);
    // LinAlg::show_matrix(layer_out1);
    // cout << endl;
    layer_out1 = ReLU.forward(layer_out1);
    // LinAlg::show_matrix(layer_out1);
    // cout << endl;
    

    vector <vector<double>>  layer_out2 = dense2.forward(layer_out1);
    double loss = Loss_Softmax.forward(layer_out2, y);
    // LinAlg::show_matrix(Loss_Softmax.softmax_output);




    //  ****************  Backward pass  ************************
    vector <vector<double>>  dL_dZ1 = Loss_Softmax.softmax_output;
    // cout << "dL_dZ1: " << dL_dZ1.size() << "x" << dL_dZ1[0].size() << endl;
    // cout << "y_true: " << y.size() << "x" << y[0].size() << endl;

    vector <vector<double>> dL_dZ2 = Loss_Softmax.backward(dL_dZ1, y);
    // cout << "dL_dZ2: " << dL_dZ2.size() << "x" << dL_dZ2[0].size() << endl;

    dense2.backward(dL_dZ2);

    // cout << "dense2.dl_dinputs: " << dense2.dL_dinputs.size() << "x" << dense2.dL_dinputs[0].size() << endl;
    // LinAlg::show_matrix(dense2.dL_dinputs);
    // cout << endl;
    
    vector <vector<double>> dL_dZ3 = ReLU.backward(dense2.dL_dinputs);
   
    // cout << "dL_dZ3: " << dL_dZ3.size() << "x" << dL_dZ3[0].size() << endl;

    dense1.backward(dL_dZ3);
    

    // cout << "dL_dW: " << dense1.dL_dW.size() << "x" << dense1.dL_dW[0].size() << endl;
    // cout << "dL_db: " << dense1.dL_db.size() << "x" << dense1.dL_db[0].size() << endl;

    

    
    learning_rate = optimizer.pre_update_params();
    // Update dense2 weights and biases
    dense2.weights = optimizer.update_weight(dense2.weights, dense2.dL_dW);
    dense2.biases = optimizer.update_weight(dense2.biases, dense2.dL_db);

// Update dense1 weights and biases
    dense1.weights = optimizer.update_weight(dense1.weights, dense1.dL_dW);
    dense1.biases = optimizer.update_weight(dense1.biases, dense1.dL_db);
    optimizer.post_update_params();

    if (k % 100 == 0) {
            cout << "Iteration: " << k << "||";
            cout << "Loss: " << loss << "||";
            cout << "Learning Rate: " << learning_rate << endl;
        }

   

    // Print first few weight updates
// cout << "Weight updates (sample):" << endl;
// for (int i = 0; i < min(3, (int)dense1.weights.size()); i++) {
//     for (int j = 0; j < min(3, (int)dense1.weights[0].size()); j++) {
//         cout << dense1.weights[i][j] << " -> " 
//              << (dense1.weights[i][j] - learning_rate * dense1.dL_dW[i][j]) << endl;
//     }
// }

// cout << "Backward pass completed successfully" << endl;
// cout << "Weights updated" << endl;
    }


    cout << endl << "Program Run Successfully!!!" << endl;


    auto end = high_resolution_clock::now();    // End timer
    duration<double> duration_sec = end - start;

    std::cout << "Execution time: " << duration_sec.count() << " seconds"<< endl;

    return 0;

}