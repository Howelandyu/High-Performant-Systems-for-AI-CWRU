#include <assert.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <string.h>
#include "mkl.h"
#include <iostream>

struct TensorDim {
    int n; // Batch size
    int c; // Channels
    int h; // Width
    int w; // Height

    // Constructor for easy initialization
    TensorDim(int batch_size, int channels, int height , int width)
        :   n(batch_size), c(channels), h(height), w(width) {}
};
// read data from file 
std::vector<double> readDataFromFile(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    double value;

    while (file >> value) {
        data.push_back(value);
    }

    return data;
}
// convert the matrices
void NCHW2HWNC(const double *nchw_data, int N, int C, int H, int W,
               double *hwnc_data) {
    // rearranging data
    for (int n = 0; n < N; n++) {
        int in_batch_offset = n * C * H * W;
        int out_batch_offset = n * C;
        for (int c = 0; c < C; ++c) {
            int in_ch_offset = c * H * W + in_batch_offset;
            int out_ch_offset = out_batch_offset + c;
            for (int h = 0; h < H; ++h) {
                int in_row_offset = h * W + in_ch_offset;
                int out_row_offset = h * C * N * W + out_ch_offset;
                for (int w = 0; w < W; ++w) {
                    int in_addr = w + in_row_offset;
                    int out_addr = out_row_offset + w * N * C;
                    hwnc_data[out_addr] = nchw_data[in_addr];
                }
            }
        }
    }
}
// matrix shift-add operation
void MatrixShiftAdd(double *base_mat, int base_no_rows, int base_no_cols, 
                    double *overlap_mat, int ov_no_rows, int ov_no_cols, 
                    int row_shift, int col_shift) {
    if (row_shift == 0 && col_shift == 0 && (base_no_rows == ov_no_rows) &&
        (base_no_cols == ov_no_cols)) {
        // normal matrix add
        cblas_daxpy(base_no_rows * base_no_cols, 1.0, overlap_mat, 1, base_mat, 1);
        return;
    }
    int rows_to_add, cols_to_add;
    int base_row_start, base_col_start;
    // overlapping row and column
    int ov_row_start, ov_col_start;
     // without padding case
    if (ov_no_rows > base_no_rows) {
        rows_to_add = base_no_rows;
        cols_to_add = base_no_cols;
        base_row_start = 0;
        base_col_start = 0;
        ov_row_start = row_shift < 0 ? -row_shift : 0;
        ov_col_start = col_shift < 0 ? -col_shift : 0;
    } else {
        rows_to_add = ov_no_rows - abs(row_shift);
        cols_to_add = ov_no_cols - abs(col_shift);
        ov_col_start = col_shift > 0 ? col_shift : 0;
        ov_row_start = row_shift > 0 ? row_shift : 0;
        base_row_start = row_shift < 0 ? -row_shift : 0;
        base_col_start = col_shift < 0 ? -col_shift : 0;
    }

    for (int r = 0; r < rows_to_add; ++r) {
        int base_mat_offset = (r + base_row_start) * base_no_cols + base_col_start;
        int overlap_mat_offset = (r + ov_row_start) * ov_no_cols + ov_col_start;
        cblas_daxpy(cols_to_add, 1.0, overlap_mat + overlap_mat_offset, 1,
                    base_mat + base_mat_offset, 1);
    }
}

bool Kn2RowConvLayer(const double *in_data, const double *filters,
                     const double *bias, TensorDim in_dim,
                     TensorDim filt_dim, int stride, int pad, int group,
                     double *output) {
    // Currently we have limited support.
    assert(group == 1);
    assert((pad == 0) || (pad == filt_dim.w / 2));
    assert(in_dim.n == 1);
    assert(filt_dim.h == filt_dim.w);
    assert(stride == 1);
    // Assuming TensorDim is a struct or class that has members: w, h, c, n
    TensorDim out_dim = {0, 0, 0, 0};
    // Calculate output width
    out_dim.w = (in_dim.w + 2 * pad - filt_dim.w) / stride + 1;
    // Calculate output height
    out_dim.h = (in_dim.h + 2 * pad - filt_dim.h) / stride + 1;
    // The number of output channels is the same as the number of filters (n)
    out_dim.c = filt_dim.n;
    // The number of output batches is the same as the number of input batches
    out_dim.n = in_dim.n;

    double *kkmc_filters = new double[filt_dim.n * filt_dim.c * filt_dim.h * filt_dim.w];
    NCHW2HWNC(filters, filt_dim.n, filt_dim.c, filt_dim.h, filt_dim.w, kkmc_filters);

    double alpha = 1.0, beta = 0.0;

    double *gemm_output = new double[out_dim.c * in_dim.h * in_dim.w];

    if (bias) {
        for (int m = 0; m < out_dim.c; ++m) {
            std::fill_n(output + m * out_dim.h * out_dim.w, out_dim.h * out_dim.w, bias[m]);
            // For batch size > 1
            for (int b = 1; b < out_dim.n; ++b) {
                std::copy(output, output + out_dim.c * out_dim.h * out_dim.w, 
                          output + b * out_dim.c * out_dim.h * out_dim.w);
            }
        }
    } else {
        memset(output, 0, out_dim.n * out_dim.c * out_dim.h * out_dim.w * sizeof(double));
    }

    for (int kr = 0; kr < filt_dim.h; kr++) {
        int row_shift = kr - filt_dim.h / 2;
        for (int kc = 0; kc < filt_dim.w; kc++) {
            int group_no = kr * filt_dim.w + kc;
            int col_shift = kc - filt_dim.w / 2;
            // Matrix dimensions - A -> mxk B -> kxn  C --> mxn
            int m = filt_dim.n;
            int k = filt_dim.c;
            int n = in_dim.h * in_dim.w;
             //# transpose status of matrices
            CBLAS_TRANSPOSE transA = CblasNoTrans;
            CBLAS_TRANSPOSE transB = CblasNoTrans;
            
            // Define leading dimensions for the matrices
            int ldA = k; // Since kkmc_filters is row-major in memory and non-transposed
            int ldB = n; // Since in_data is row-major in memory and non-transposed
            int ldC = n; // Since gemm_output is row-major in memory
            
            // do operations
            cblas_dgemm(CblasRowMajor, transA, transB, m, n, k, alpha, 
                kkmc_filters + group_no * m * k, ldA, 
                in_data, ldB, beta, 
                gemm_output, ldC);

            for (int omap = 0; omap < filt_dim.n; omap++) {
                MatrixShiftAdd(output + omap * out_dim.h * out_dim.w,
                               out_dim.h, out_dim.w,
                               gemm_output + omap * in_dim.h * in_dim.w,
                               in_dim.h, in_dim.w, row_shift, col_shift);
            }
        }
    }

    delete[] kkmc_filters;
    delete[] gemm_output;
    // Free the USM memory
    return true;
}

int main() {
    // Set input dimensions: height = 32, width = 32, channels = 1, batch size = 1
    TensorDim in_dim = {1, 1, 32, 32};
    // Set filter dimensions: kernel height = 2, kernel width = 2, in channels = 1, out channels = 64
    TensorDim filt_dim = {64, 1, 2, 2};
    
    // Convolution parameters
    int stride = 1;
    int pad = 1;
    int group = 1;
    
    // Read input data from file
    std::string input_filename = "flat_test_x.txt";
    std::vector<double> inputData = readDataFromFile(input_filename);
    
    
    // Define filters and bias
    std::vector<double> filters(filt_dim.w * filt_dim.h * filt_dim.c * filt_dim.n);
    std::vector<double> bias(filt_dim.n);
    
    // Initialize all weights to 0.1
    std::fill(filters.begin(), filters.end(), 0.1);
    
    // Initialize all biases to 0
    std::fill(bias.begin(), bias.end(), 0.0);
    
    // Allocate memory for the output
    TensorDim out_dim = {0, 0, 0, 0};
    out_dim.w = (in_dim.w + 2 * pad - filt_dim.w) / stride + 1;
    out_dim.h = (in_dim.h + 2 * pad - filt_dim.h) / stride + 1;
    out_dim.c = filt_dim.n;
    out_dim.n = in_dim.n;
    std::vector<double> outputData(out_dim.w * out_dim.h * out_dim.c * out_dim.n);
    
    // Perform the convolution
    if (!Kn2RowConvLayer(inputData.data(), filters.data(), bias.data(), in_dim, filt_dim, stride, pad, group, outputData.data())) {
        std::cerr << "Convolution failed" << std::endl;
        return EXIT_FAILURE;
    }

    // Write the output data to a file
    std::ofstream outputFile("kn2row_conv_output.txt");
    for (const auto& value : outputData) {
        outputFile << value << "\n";
    }

    return EXIT_SUCCESS;
}