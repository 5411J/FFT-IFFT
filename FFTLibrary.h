#ifndef FFT_LIBRARY_H
#define FFT_LIBRARY_H

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

namespace FFTLibrary {

    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X); // Y = fft(X)
    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X, int n); // Y = fft(X, n)
    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X, int n, int dim); // Y = fft(X, n, dim)
    Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X);
    Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X, int n); // Y = ifft(X, n)
    Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X, int n, int dim); // Y = ifft(X, n, dim)

} // namespace FFTLibrary

#endif // FFT_LIBRARY_H
