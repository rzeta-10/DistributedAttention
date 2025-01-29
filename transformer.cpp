#include <bits/stdc++.h>
using namespace std;

vector<vector<double>> concatenate_heads(vector<vector<double>>&, int, int, int);
void add_bias(vector<vector<double>>&, vector<double>&);
vector<vector<double>> genRandomMatrix(int, int);
vector<vector<double>> split_heads(vector<vector<double>>&, int, int);
vector<vector<double>> matmul(vector<vector<double>>&, vector<vector<double>>&);
void softmax_rows(vector<vector<double>>&);
vector<vector<double>> softmax(vector<vector<double>>&);
vector<vector<double>> matrixTranspose(vector<vector<double>>&);
vector<vector<double>> read_data(string, int, int);
vector<vector<double>> get_positional_encoding(int, int);

class MultiHeadAttention {
    public:
        int d_model, num_heads, d_key, d_value;
        vector<vector<double>> WQ, WK, WV, WO;

        MultiHeadAttention(int d_model, int num_heads) : d_model(d_model), num_heads(num_heads) {
            assert(d_model % num_heads == 0);
            this->d_model = d_model;
            this->num_heads = num_heads;
            this->d_key = d_model / num_heads;
            this->d_value = d_model / num_heads;
            this->WQ = genRandomMatrix(d_model, d_model);
            this->WK = genRandomMatrix(d_model, d_model);
            this->WV = genRandomMatrix(d_model, d_model);
            this->WO = genRandomMatrix(d_model, d_model);
        }

        vector<vector<double>> forward(vector<vector<double>>& x) {
            int seq_len = x.size();

            vector<vector<double>> Q = matmul(x, WQ);
            vector<vector<double>> K = matmul(x, WK);
            vector<vector<double>> V = matmul(x, WV);

            vector<vector<double>> Q_heads = split_heads(Q, num_heads, d_key);
            vector<vector<double>> K_heads = split_heads(K, num_heads, d_key);
            vector<vector<double>> V_heads = split_heads(V, num_heads, d_value);

            vector<vector<double>> attention_heads;
            attention_heads.reserve(num_heads * seq_len);

            for (int i = 0; i < num_heads; i++) {
                vector<vector<double>> Q_head, K_head, V_head;
                Q_head.reserve(seq_len);
                K_head.reserve(seq_len);
                V_head.reserve(seq_len);

                for (int i = 0; i < seq_len; i++) {
                    Q_head.push_back(Q_heads[i * num_heads + i]);
                    K_head.push_back(K_heads[i * num_heads + i]);
                    V_head.push_back(V_heads[i * num_heads + i]);
                }

                vector<vector<double>> K_head_T = matrixTranspose(K_head);
                vector<vector<double>> attention_scores = matmul(Q_head, K_head_T);

                double scale = sqrt(d_key);
                for (auto& row : attention_scores) {
                    for (auto& val : row) {
                        val /= scale;
                    }
                }

                softmax_rows(attention_scores);

                vector<vector<double>> attention_output = matmul(attention_scores, V_head);

                for (int i = 0; i < seq_len; i++) {
                    attention_heads.emplace_back(attention_output[i]);
                }
            }

            vector<vector<double>> concat = concatenate_heads(attention_heads, num_heads, seq_len, d_value);

            vector<vector<double>> output = matmul(concat, WO);

            return output;
        }
};

class FeedForward {
    public:
        int d_model, ff_dim;
        vector<vector<double>> W1, W2;
        vector<double> b1, b2;

        FeedForward(int d_model, int ff_dim) : d_model(d_model), ff_dim(ff_dim) {
            this->W1 = genRandomMatrix(d_model, ff_dim);
            this->b1 = vector<double>(ff_dim, 0.0f);
            this->W2 = genRandomMatrix(ff_dim, d_model);
            this->b2 = vector<double>(d_model, 0.0f);
        }

        vector<vector<double>> forward(vector<vector<double>>& x) {
            // Layer 1
            vector<vector<double>> h1 = matmul(x, W1);
            add_bias(h1, b1);

            // ReLU activation
            for (auto& row : h1) {
                for (auto& val : row) {
                    val = max(0.0, val);
                }
            }
            vector<vector<double>> h2 = matmul(h1, W2);
            add_bias(h2, b2);

            return h2;
        }
};

vector<vector<double>> concatenate_heads(vector<vector<double>>& x, int num_heads, int seq_len, int d_value) {
    vector<vector<double>> X(seq_len, vector<double>(num_heads * d_value, 0.0f));
    for (int i = 0; i < num_heads; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < d_value; k++) {
                X[j][i * d_value + k] = x[i * seq_len + j][k];
            }
        }
    }
    return X;
}

void add_bias(vector<vector<double>>& x, vector<double>& b) {
    assert(x[0].size() == b.size());
    for (auto& row : x) {
        for (int i = 0; i < row.size(); i++) {
            row[i] += b[i];
        }
    }
}

vector<vector<double>> genRandomMatrix(int rows, int cols) {
    vector<vector<double>> matrix(rows, vector<double>(cols, 0.0f));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
    return matrix;
}

vector<vector<double>> split_heads(vector<vector<double>>& x, int num_heads, int d_head) {
    int seq_len = x.size();
    int d_model = x[0].size();
    vector<vector<double>> X_split(seq_len * num_heads, vector<double>(d_head, 0.0f));

    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_head; ++j) {
                X_split[h * seq_len + i][j] = x[i][h * d_head + j];
            }
        }
    }
    return X_split;
}

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

void softmax_rows(vector<vector<double>>& a) {
    for (auto& row : a) {
        double max_val = *max_element(row.begin(), row.end());
        double sum = 0.0f;
        for (auto& val : row) {
            val = exp(val - max_val);
            sum += val;
        }
        for (auto& val : row) {
            val /= sum;
        }
    }
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

vector<vector<double>> get_positional_encoding(int sequence_length, int d_model) {
    vector<vector<double>> positional_encodings(sequence_length, vector<double>(d_model, 0.0f));
    for (int pos = 0; pos < sequence_length; pos++) {
        for (int i = 0; i < d_model; i++) {
            if (i % 2 == 0) {
                positional_encodings[pos][i] = sin(pos / pow(10000, (double)i / d_model));
            } else {
                positional_encodings[pos][i] = cos(pos / pow(10000, (double)(i - 1) / d_model));
            }
        }
    }
    return positional_encodings;
}

class EncoderLayer {
    public:
        MultiHeadAttention mha;
        FeedForward ff;

        EncoderLayer(int d_model, int num_heads, int ff_dim) : mha(d_model, num_heads), ff(d_model, ff_dim) {}

        vector<vector<double>> forward(vector<vector<double>>& x) {
            vector<vector<double>> attn_output = mha.forward(x);
            vector<vector<double>> ff_output = ff.forward(attn_output);
            return ff_output;
        }
};

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
    cout << "Data read successfully with size "<< input_vectors.size() << endl;

    vector<double> sample = input_vectors[0];
    vector<vector<double>> sample_vector(sequence_length, vector<double>(embed_dim, 0.0f));

    for(int i = 0; i< sequence_length; i++){
        for(int j = 0; j < embed_dim; j++){
            sample_vector[i][j] = sample[i*embed_dim + j];
        }
    }

    vector<vector<double>> positional_encodings = get_positional_encoding(sequence_length, d_model);
    
    for(int i = 0; i < sequence_length; i++){
        for(int j = 0; j < d_model; j++){
            sample_vector[i][j] += positional_encodings[i][j];
        }
    }

    EncoderLayer encoder_layer(d_model, num_heads, ff_dim);
    vector<vector<double>> encoder_output = encoder_layer.forward(sample_vector);

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