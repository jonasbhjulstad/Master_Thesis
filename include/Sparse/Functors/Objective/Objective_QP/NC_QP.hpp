#ifndef FIPOPT_NC_QP_HPP
#define FIPOPT_NC_QP_HPP
#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Objective/Objective_Memoized/Objective_Memoized.hpp>
#include <Common/EigenDataTypes.hpp>
#include <fstream>

// Nonlinearly constrained QP

namespace FIPOPT::Sparse
{

    struct NC_QP : public objective<NC_QP, 2,1,0>
    {
        constexpr static int Nx = 2;
        constexpr static int Ng = 1;
        constexpr static int Nh = 0;
        using Diag_x = Eigen::DiagonalMatrix<double, Nx>;


    protected:
        const dVec x_ub_ = dVec::Constant(1e20);
        const dVec x_lb_ = dVec::Constant(-1e20);

    public:

        template <typename BaseType>
        inline Val operator()(const BaseType &x)
        {
            return .5 * x.transpose()*x;
        }

        template <typename BaseType>
        inline dVec Eval_grad(const BaseType &x)
        {
            return x;
        }

        template <typename BaseType>
        inline spMat Eval_hessian_f(const BaseType &x)
        {
            return spMat::Identity();
        }

        template <typename BaseType>
        inline dVec Eval_g(const BaseType &x)
        {
            return dVec(x.transpose()*x-1);
        }

        template <typename BaseType>
        inline spMat Eval_grad_g(const BaseType &x)
        {
            return 2*x.transpose();
        }

        template <typename BaseType_x, typename BaseType_lbd>
        inline spMat Eval_hessian_g(const BaseType_x &x, const BaseType_lbd &lbd_g)
        {
            return lbd_g.value()*2*spMat::Identity();
        }

        inline dVec Get_x_lb()
        {
            return x_lb_;
        }

        inline dVec Get_x_ub()
        {
            return x_ub_;
        }
    };

}

#endif