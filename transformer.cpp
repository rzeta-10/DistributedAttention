#include <bits/stdc++.h>
using namespace std;

vector<vector<double>> matmul(vector<vector<double>>& a, vector<vector<double>>& b) {
    int n = a.size(), m = a[0].size(), p = b[0].size();
    vector<vector<double>> c(n, vector<double>(p, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < m; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

vector<vector<double>> softmax(vector<vector<double>>& a){
    for(auto& row : a){
        double max_val = *max_element(row.begin(), row.end());
        double sum = 0.0f;
        for(auto& val : row){
            val = exp(val - max_val);
            sum += val;
        }
        for(auto& val : row){
            val /= sum;
        }
    }
    return a;
}

vector<vector<double>> matrixTranspose(vector<vector<double>>& a){
    vector<vector<double>> b(a[0].size(), vector<double>(a.size(), 0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            b[j][i] = a[i][j];
        }
    }
    return b;
}
vector<vector<double>> read_data(string filename, int sequence_length, int embed_dim) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "File not found!" << endl;
        return {};
    }
    vector<vector<double>> data;
    string line;
    int expected_dim = sequence_length * embed_dim;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> vec(embed_dim, 0);
        for (int i = 0; i < embed_dim; i++) {
            ss >> vec[i];
        }
        if(vec.size() != embed_dim){
            cout << "Mismatch in sample size. Skipping sample" << endl;
            continue;
        }
        data.push_back(vec);
    }
    return data;
}

int main() {

    // transformer parameters
    int d_model = 100;
    int embed_dim = 100;
    int sequence_length = 20;
    int num_heads = 10;
    int ff_dim = 256;

    string input_file = "dataset_vectors.txt";

    cout << "Loading the data from the file..." << endl;

    vector<vector<double>> input_vectors = read_data(input_file, sequence_length, embed_dim);
    if(input_vectors.empty()){
        cout << "No data found in the file. Exiting..." << endl;
        return 1;
    }
    cout << "Data read successfully!" << endl;

    vector<vector<double>> a = {{1, 2}, {3, 4}};
    vector<vector<double>> b = {{5, 6}, {7, 8}};
    vector<vector<double>> b_T = matrixTranspose(b);
    vector<vector<double>> c = matmul(a, b_T);
    for (int i = 0; i < c.size(); i++) {
        for (int j = 0; j < c[0].size(); j++) {
            cout << c[i][j] / sqrt(2) << " ";
        }
        cout << endl;
    }
    vector<vector<double>> softmax_result = softmax(c);
    for (const auto& row : softmax_result) {
        for (const auto& val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}