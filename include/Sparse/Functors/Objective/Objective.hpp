#ifndef FIPOPT_OBJECTIVE_Sparse_HPP
#define FIPOPT_OBJECTIVE_Sparse_HPP
#include <Common/EigenDataTypes.hpp>
#include <Common/Utils/Eigen_Utils.hpp>
#include <vector>
namespace FIPOPT::Sparse
{

    template <typename Derived>
    struct objective
    {

        void Initialize(const int &Nx, const int &Ng, const int &Nh)
        {
            Nx_ = Nx;
            Ng_ = Ng;
            Nh_ = Nh;
            std::vector<Triplet> T;
            grad_cI_.resize(Ng_ + 2 * Nx_, Nx_);
            for (int i = 0; i < Nx_; i++)
            {
                T.push_back(Triplet(Ng_ + i, i, -1.));
                T.push_back(Triplet(Ng_ + Nx_ + i, i, 1.));
            }
            grad_cI_.setFromTriplets(T.begin(), T.end());
        }

        inline Val operator()(const MatrixBase<dVec> &x)
        {
            return static_cast<Derived *>(this)->operator()(x);
        }

        inline dVec Eval_grad(const MatrixBase<dVec> &x)
        {
            return static_cast<Derived *>(this)->Eval_grad(x);
        }

        inline spMat Eval_hessian_f(const MatrixBase<dVec> &x)
        {
            return static_cast<Derived *>(this)->Eval_hessian_f(x);
        }

        inline dVec Eval_cE(const MatrixBase<dVec> &x)
        {
            if (Nh_ == 0)
            {
                return dVec(0);
            }
            else
            {
                return static_cast<Derived *>(this)->Eval_h(x);
            }
        }

        inline spMat Eval_grad_cE(const MatrixBase<dVec> &x)
        {
            if (Nh_ == 0)
            {
                return spMat(0, Nx_);
            }
            else
            {
                return static_cast<Derived *>(this)->Eval_grad_h(x);
            }
        }

        inline spMat Eval_hessian_cE(const MatrixBase<dVec> &x, const MatrixBase<dVec> &lbd)
        {
            if (Nh_ == 0)
            {
                return spMat(0, Nx_);
            }
            else
            {
                return static_cast<Derived *>(this)->Eval_hessian_h(x, lbd);
            }
        }

        inline spMat Eval_hessian(const MatrixBase<dVec> &x, const MatrixBase<dVec> &lbd)
        {
            return Eval_hessian_f(x) + Eval_hessian_cE(x, lbd);
        }

        inline dVec Eval_cI(const MatrixBase<dVec> &x)
        {
            dVec cI(Ng_ + 2 * Nx_);
            if (Ng_ > 0)
            {
                cI.topRows(Ng_) = dVec(static_cast<Derived *>(this)->Eval_g(x));
            }
            cI.middleRows(Ng_, Nx_) = Get_x_ub() - x;
            cI.bottomRows(Nx_) = x - Get_x_lb();

            return cI.sparseView();
        }

        inline spMat Eval_grad_cI(const MatrixBase<dVec> &x)
        {
            if (Ng_ > 0)
            {
                spMat res = static_cast<Derived *>(this)->Eval_grad_g(x);
                block_assign(grad_cI_, res);
            }
            return grad_cI_;
        }

        inline spMat Eval_hessian_cI(const MatrixBase<dVec> &x, const MatrixBase<dVec> &lbd)
        {
            if (Ng_ == 0)
            {
                return spMat(Nx_, Nx_);
            }
            else
            {
                return static_cast<Derived *>(this)->Eval_hessian_g(x, lbd);
            }
        }

        inline dVec Get_x_lb()
        {
            return static_cast<Derived *>(this)->Get_x_lb();
        }
        inline dVec Get_x_ub()
        {
            return static_cast<Derived *>(this)->Get_x_ub();
        }

        inline int Get_Nx()
        {
            return Nx_;
        }
        inline int Get_Ng()
        {
            return Ng_;
        }
        inline int Get_Nh()
        {
            return Nh_;
        }

    protected:
        int Nx_, Ng_, Nh_;

    private:
        dVec cI_;
        spMat grad_cI_;
    };
}
#endif