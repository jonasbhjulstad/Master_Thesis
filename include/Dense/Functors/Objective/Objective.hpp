#ifndef FIPOPT_OBJECTIVE_DENSE_HPP
#define FIPOPT_OBJECTIVE_DENSE_HPP
#include <Common/EigenDataTypes.hpp>
#include <vector>
namespace FIPOPT::Dense
{

    template <typename Derived, int nx, int ng, int nh>
    struct objective
    {
        static constexpr int Nx = nx;
        static constexpr int Ng = ng;
        static constexpr int Nh = nh;
        // Dense types
        using Mat_x = Eigen::Matrix<double, Nx, Nx>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Vec_g = Eigen::Matrix<double, Ng, 1>;
        using Mat_cI = Eigen::Matrix<double, std::max(-1, Ng + 2 * Nx), Nx>;
        using Vec_cI = Eigen::Matrix<double, std::max(-1, Ng + 2 * Nx), 1>;
        using Mat_cE = Eigen::Matrix<double, Nh, Nx>;
        using Vec_cE = Eigen::Matrix<double, Nh, 1>;

        private:
        Vec_cI cI_;
        Mat_cI grad_cI_;

        public:
        objective()
        {
            grad_cI_.middleRows(Ng, Nx) = -Mat_x::Identity();
            grad_cI_.bottomRows(Nx) = Mat_x::Identity();
        }

        inline Val operator()(const MatrixBase<Vec_x> &x)
        {
            return static_cast<Derived *>(this)->operator()(x);
        }

        inline Vec_x Eval_grad(const MatrixBase<Vec_x> &x)
        {
            return static_cast<Derived *>(this)->Eval_grad(x);
        }

        inline Mat_x Eval_hessian_f(const MatrixBase<Vec_x> &x)
        {
            return static_cast<Derived *>(this)->Eval_hessian_f(x);
        }

        inline Vec_cE Eval_cE(const MatrixBase<Vec_x> &x)
        {
            if constexpr (Nh == 0)
            {
                return Vec_cE::Zero();
            }
            else
            {
                return static_cast<Derived *>(this)->Eval_h(x);
            }
        }

        inline Mat_cE Eval_grad_cE(const MatrixBase<Vec_x> &x)
        {
            if constexpr (Nh == 0)
            {
                return Mat_cE::Zero();
            }
            else
            {
                return static_cast<Derived *>(this)->Eval_grad_h(x);
            }
        }

        inline Mat_x Eval_hessian_cE(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_cE> &lbd)
        {
            if constexpr (Nh == 0)
            {
                return Mat_x::Zero();
            }
            else
            {
                return static_cast<Derived *>(this)->Eval_hessian_h(x, lbd);
            }
        }

        inline Mat_x Eval_hessian(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_cE> &lbd)
        {
            return Eval_hessian_f(x) + Eval_hessian_cE(x, lbd);
        }

        inline Vec_cI Eval_cI(const MatrixBase<Vec_x> &x)
        {
            if constexpr (Ng > 0)
                cI_.head(Ng) = static_cast<Derived *>(this)->Eval_g(x);
            cI_.segment(Ng, Nx) = Get_x_ub() - x;
            cI_.tail(Nx) = x - Get_x_lb();
            return cI_;
        }

        inline Mat_cI Eval_grad_cI(const MatrixBase<Vec_x> &x)
        {
            if constexpr (Ng > 0)
                grad_cI_.topLeftCorner(Ng, Nx) = static_cast<Derived *>(this)->Eval_grad_g(x);
            return grad_cI_;
        }

        inline Mat_x Eval_hessian_cI(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_g> &lbd)
        {
            if constexpr (Ng == 0)
            {
                return Mat_x::Zero();
            }
            else
            {
                return static_cast<Derived *>(this)->Eval_hessian_g(x, lbd);
            }
        }

        inline Vec_x Get_x_lb()
        {
            return static_cast<Derived *>(this)->Get_x_lb();
        }
        inline Vec_x Get_x_ub()
        {
            return static_cast<Derived *>(this)->Get_x_ub();
        }
    };

}
#endif