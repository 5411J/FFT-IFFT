#include <iostream>
#include "FFTLibrary.h"

int main() {
    Eigen::MatrixXd X(4, 3);
    X << 1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12;

    // Case 1: Y = fft(X)
    Eigen::MatrixXcd Y1 = FFTLibrary::fft(X);
    std::cout << "fft(X):\n" << Y1 << "\n\n";

    // Case 2: Y = fft(X, n)
    Eigen::MatrixXcd Y2 = FFTLibrary::fft(X, 3);
    std::cout << "fft(X, n):\n" << Y2 << "\n\n";

    // Case 3: Y = fft(X, n, dim)
    Eigen::MatrixXcd Y3 = FFTLibrary::fft(X, 3, 2);
    std::cout << "fft(X, n, 1):\n" << Y3 << "\n\n";

    Eigen::MatrixXcd Y4 = FFTLibrary::ifft(X);
    std::cout << "fft(X):\n" << Y4 << "\n\n";

    // Case 2: Y = fft(X, n)
    Eigen::MatrixXcd Y5 = FFTLibrary::ifft(X, 4);
    std::cout << "fft(X, n):\n" << Y5 << "\n\n";

    // Case 3: Y = fft(X, n, dim)
    Eigen::MatrixXcd Y6 = FFTLibrary::ifft(X, 4, 2);
    std::cout << "fft(X, n, 1):\n" << Y6 << "\n\n";

    std::vector <float>r1 = {
            1,1,1
    };
    std::vector<float>r2 = {
            1,1,0,0,0,1,1
    };
    std::vector<float>result = FFTLibrary::convolve(r1,r2);
    std::cout << "convolve(r1,r2)" << std::endl;
    for(auto element : result) {
        std::cout << element << " ";
    }
    return 0;
}
