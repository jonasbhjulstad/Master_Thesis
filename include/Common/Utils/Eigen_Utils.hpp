#ifndef FIPOPT_EIGEN_UTILS_HPP
#define FIPOPT_EIGEN_UTILS_HPP
#include <Common/EigenDataTypes.hpp>
#include <limits>
#include <cmath>
#include <algorithm>
namespace FIPOPT
{
    template <typename T>
    inline double l1_norm(const SparseMatrixBase<T> &M)
    {
        return (Eigen::RowVectorXd::Ones(M.rows()) * M.cwiseAbs()).maxCoeff();
    }
    template <typename T>
    inline double l1_norm(const MatrixBase<T> &M)
    {
        return M.template lpNorm<1>();
    }

    template <typename T>
    inline double linf_norm(const SparseMatrixBase<T> &M)
    {
        spMat mat = M;
        double res = -std::numeric_limits<double>::infinity();
        for (int k = 0; k < mat.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
            {
                res = std::max(res, std::abs(it.value()));
            }
        return res;
    }

    template <typename T>
    inline double linf_norm(const MatrixBase<T> &M)
    {
        return M.template lpNorm<Eigen::Infinity>();
    }

    inline double l2_norm(const spVec& V)
    {
        double square_sum = 0;
        for (spVec::InnerIterator it(V); it; ++it)
        {
            square_sum += it.value()*it.value();
        }
        return sqrt(square_sum);
    }



    template <typename T>
    inline bool any_nan(const MatrixBase<T> &M)
    {
        return M.array().isNaN().any();
    }

    template <typename T>
    inline bool any_nan(const SparseMatrixBase<T> &M)
    {
        spMat mat = M;
        for (int k = 0; k < mat.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
            {
                if (std::isnan(it.value()))
                    return true;
            }
        return false;
    }

    template <typename T>
    inline void set_diagonal(SparseMatrixBase<T> &M, const double &val)
    {
        for (int i = 0; i < M.rows(); i++)
        {
            M.insert(i, i) = val;
        }
    }

    inline void set_diagonal(spMat &M, const double &val, const int &rs, const int &cs, const double &N)
    {
        for (int i = 0; i < N; i++)
        {
            M.insert(rs + i, cs + i) = val;
        }
    }

    template <typename T>
    inline void set_vector(spMat &M, SparseMatrixBase<T> &D, const double &offset)
    {
        for (typename Eigen::SparseMatrix<double>::InnerIterator it(D); it; ++it)
        {
            M.insert(offset + it.index(), 0) = it.value();
        }
    }

    inline void set_diagonal(spMat &M, spVec &diag)
    {
        for (spVec::InnerIterator it(diag); it; ++it)
        {
            M.insert(it.index(), it.index()) = it.value();
        }
    }

    inline bool any_smaller(spVec &&M, const double &coeff)
    {
        for (int k = 0; k < M.outerSize(); ++k)
            for (spVec::InnerIterator it(M, k); it; ++it)
            {
                if (it.value() < coeff)
                    return true;
            }
        return false;
    }

    inline bool all_larger(const spVec &&M, const double &coeff)
    {
        for (int k = 0; k < M.outerSize(); ++k)
            for (spVec::InnerIterator it(M, k); it; ++it)
            {
                if (it.value() < coeff)
                    return false;
            }
        return true;
    }
    template <typename T>
    inline void block_assign(spMat &M, SparseMatrixBase<T> &data, const int &row_offset = 0, const int &col_offset = 0)
    {
        spMat D = data;
        for (int k = 0; k < D.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(D, k); it; ++it)
            {
                M.insert(row_offset + it.row(), col_offset + it.col()) = it.value();
            }
    }

    inline std::vector<Eigen::Triplet<double>> to_triplets(Eigen::SparseMatrix<double> &M)
    {
        std::vector<Eigen::Triplet<double>> T;
        for (int i = 0; i < M.outerSize(); i++)
            for (typename Eigen::SparseMatrix<double>::InnerIterator it(M, i); it; ++it)
                T.emplace_back(it.row(), it.col(), it.value());
        return T;
    }

}

#endif