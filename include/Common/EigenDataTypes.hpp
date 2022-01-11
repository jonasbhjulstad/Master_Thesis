#ifndef FIPOPT_EIGENDATATYPES_HPP
#define FIPOPT_EIGENDATATYPES_HPP
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <type_traits>

using Val = Eigen::Matrix<double, 1, 1>;
using spVal = Eigen::SparseVector<double>;
using spVec = Eigen::SparseVector<double>;
using spMat = Eigen::SparseMatrix<double>;
using dVec = Eigen::VectorXd;
using dMat = Eigen::MatrixXd;
using Eigen::MatrixBase;
using Eigen::SolverBase;
using Eigen::SparseMatrixBase;
using Eigen::SparseVector;
using Eigen::SparseMatrix;
using Triplet = Eigen::Triplet<double>;

#endif
