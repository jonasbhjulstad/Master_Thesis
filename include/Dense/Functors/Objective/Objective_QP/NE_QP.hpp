#ifndef FIPOPT_NC_QP_HPP
#define FIPOPT_NC_QP_HPP
#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Objective/Objective_Memoized/Objective_Memoized.hpp>
#include <Common/EigenDataTypes.hpp>
#include <fstream>

namespace FIPOPT::Dense
{

    struct NC_QP : public objective<NC_QP, 2,0,1>
    {
        constexpr static int Nx = 2;
        constexpr static int Ng = 0;
        constexpr static int Nh = 1;
        using Diag_x = Eigen::DiagonalMatrix<double, Nx>;
        using Mat_x = Eigen::Matrix<double, Nx, Nx>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Mat_h = Eigen::Matrix<double, Nh, Nx>;
        using Vec_h = Eigen::Matrix<double, Nh, 1>;
        using Mat_g = Eigen::Matrix<double, Ng, Nx>;
        using Vec_g = Eigen::Matrix<double, Ng, 1>;

    protected:
        const Vec_x x_ub_ = Vec_x::Constant(1e20);
        const Vec_x x_lb_ = Vec_x::Constant(-1e20);
        Mat_x Q_;

    public:

        NC_QP()
        {
            Q_ << 10, 0, 0, 1;
        }

        template <typename BaseType>
        inline Val operator()(const BaseType &x)
        {
            return .5 * x.transpose()*Q_*x;
        }

        template <typename BaseType>
        inline Vec_x Eval_grad(const BaseType &x)
        {
            return Q_*x;
        }

        template <typename BaseType>
        inline Mat_x Eval_hessian_f(const BaseType &x)
        {
            return Q_;
        }

        template <typename BaseType>
        inline Vec_h Eval_h(const BaseType &x)
        {
            return x.transpose()*x - Vec_h::Constant(1.);
        }

        template <typename BaseType>
        inline Mat_h Eval_grad_h(const BaseType &x)
        {
            return 2*x.transpose();
        }

        template <typename BaseType_x, typename BaseType_lbd>
        inline Mat_x Eval_hessian_h(const BaseType_x &x, const BaseType_lbd &lbd_h)
        {
            return lbd_h.value()*2*Mat_x::Identity();
        }

        inline Vec_x Get_x_lb()
        {
            return x_lb_;
        }

        inline Vec_x Get_x_ub()
        {
            return x_ub_;
        }
    };

}

#endif