#ifndef FIPOPT_OBJECTIVE_OBSERVER_HPP
#define FIPOPT_OBJECTIVE_OBSERVER_HPP
#include <Common/EigenDataTypes.hpp>
#include <string>

namespace FIPOPT::Dense
{
    template <typename Derived, int Nx, int Ng, int Nh>
    struct objective_observer
    {
        // Dense types
        using Mat_x = Eigen::Matrix<double, Nx, Nx>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Mat_h = Eigen::Matrix<double, Nh, Nx>;
        using Vec_h = Eigen::Matrix<double, Nh, 1>;
        using Mat_g = Eigen::Matrix<double, Ng, Nx>;
        using Vec_g = Eigen::Matrix<double, Ng, 1>;
        using Base = Derived;

        inline void Eval_f(const MatrixBase<Vec_x> &x, const MatrixBase<Val> &res)
        {
            static_cast<Derived *>(this)->Eval_f(x, res);
        }

        inline void Eval_grad(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_x> &res)
        {
            static_cast<Derived *>(this)->Eval_grad(x, res);
        }

        inline void Eval_hessian_f(const MatrixBase<Vec_x> &x, const MatrixBase<Mat_x> &res)
        {
            static_cast<Derived *>(this)->Eval_hessian_f(x, res);
        }

        inline void Eval_h(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_h> &res)
        {
            static_cast<Derived *>(this)->Eval_h(x, res);
        }

        inline void Eval_grad_h(const MatrixBase<Vec_x> &x, const MatrixBase<Mat_h> &res)
        {
            static_cast<Derived *>(this)->Eval_grad_h(x, res);
        }

        inline void Eval_hessian_h(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_h> &lbd, const MatrixBase<Mat_x> &res)
        {
            static_cast<Derived *>(this)->Eval_hessian_h(x, lbd, res);
        }

        inline void Eval_g(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_g> &res)
        {
            static_cast<Derived *>(this)->Eval_g(x, res);
        }

        inline void Eval_grad_g(const MatrixBase<Vec_x> &x, const MatrixBase<Mat_g> &res)
        {
            static_cast<Derived *>(this)->Eval_grad_g(x, res);
        }

        inline void Eval_hessian_g(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_g> &lbd_g, const MatrixBase<Mat_x> &res)
        {
            static_cast<Derived *>(this)->Eval_hessian_g(x, lbd_g, res);
        }
    };
}

#endif