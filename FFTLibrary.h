#ifndef FFT_LIBRARY_H
#define FFT_LIBRARY_H

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <vector>

namespace FFTLibrary {

    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X); // Y = fft(X)
    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X, int n); // Y = fft(X, n)
    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X, int n, int dim); // Y = fft(X, n, dim)
    Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X);// Y= ifft(X)
    Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X, int n); // Y = ifft(X, n)
    Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X, int n, int dim); // Y = ifft(X, n, dim)
    std::vector<float> convolve(std::vector<float>& r1, std::vector<float>& r2);

} // namespace FFTLibrary

#endif // FFT_LIBRARY_H
