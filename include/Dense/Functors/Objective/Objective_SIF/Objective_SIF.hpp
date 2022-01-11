#ifndef FIPOPT_OBJECTIVE_SIF_DENSE_HPP
#define FIPOPT_OBJECTIVE_SIF_DENSE_HPP
#include <iostream>
#include <string>
#include <vector>
#include <Common/Cutest.hpp>
#include <Dense/Functors/Objective/Objective_Memoized/Objective_Memoized.hpp>
#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Objective/Objective_Journalist/Objective_Journalist.hpp>
#include <Common/EigenDataTypes.hpp>

namespace FIPOPT::Dense
{
    template <int Nx, int Ng, int Nh, template <class, int, int, int> typename Base = objective>
    struct objective_SIF_base : public Base<objective_SIF_base<Nx, Ng, Nh, Base>, Nx, Ng, Nh>
    {
        using Mat_x = Eigen::Matrix<double, Nx, Nx>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Mat_h = Eigen::Matrix<double, Nh, Nx>;
        using Vec_h = Eigen::Matrix<double, Nh, 1>;
        using Mat_g = Eigen::Matrix<double, Ng, Nx>;
        using Vec_g = Eigen::Matrix<double, Ng, 1>;
        using Vec_c = Eigen::Matrix<double, Ng + Nh, 1>;
        using Mat_c = Eigen::Matrix<double, Ng + Nh, Nx>;
        using base_t = Base<objective_SIF_base<Nx, Ng, Nh, Base>, Nx, Ng, Nh>;

    private:
        void Initialize()
        {
            int dummy;
            FORTRAN_open(&funit, &fPath_[0], &ierr);

            if constexpr (Nc > 0)
            {
                int e_order, l_order, v_order;
                // Equalities should appear before inequalities:
                e_order = 1;
                // No imposed order on linear/nonlinear constraints
                l_order = 0;
                v_order = 0;
                int nnzj_, nnzh_;

                CUTEST_cdimen(&status, &funit, &Nx_, &Nc_);
                CUTEST_csetup(&status, &funit, &iout, &io_buffer,
                              &Nx_, &Nc_, x0_.data(), x_lb_.data(), x_ub_.data(),
                              c_.data(), lc_.data(), uc_.data(),
                              eqn_type_, eqn_linear_, &e_order, &l_order, &v_order);

                CUTEST_cdimsj(&status, &nnzj_);
                CUTEST_cdimsh(&status, &nnzh_);
            }
            else
            {
                CUTEST_udimen(&status, &funit, &Nx_);
                CUTEST_usetup(&status, &funit, &iout, &io_buffer, &Nx_, x0_.data(), x_lb_.data(), x_ub_.data());
            }
        }

    public:
        objective_SIF_base(const std::string &fPath) : fPath_(fPath)
        {
            Initialize();
        }
        template <typename Observer>
        objective_SIF_base(const std::string &fPath, Observer& obs): fPath_(fPath), base_t(obs) 
        {
            Initialize();
        }

        template <typename BaseType>
        inline Val operator()(const BaseType &x)
        {
            Val f;
            CUTEST_ufn(&status, &Nx_, Vec_x(x).data(), f.data());
            return f;
        }

        template <typename BaseType>
        inline Vec_x Eval_grad(const BaseType &x)
        {
            Vec_x grad_f;
            CUTEST_ugr(&status, &Nx_, Vec_x(x).data(), grad_f.data());
            return grad_f;
        }

        template <typename BaseType>
        inline Mat_x Eval_hessian_f(const BaseType &x)
        {
            Mat_x hessian_f_;
            CUTEST_udh(&status, &Nx_, Vec_x(x).data(), &Nx_, hessian_f_.data());
            CUTEST_udh(&status, &Nx_, Vec_x(x).data(), &Nx_, hessian_f_.data());
            return hessian_f_;
        }

        template <typename BaseType>
        inline Vec_h Eval_h(const BaseType &x)
        {
            double dummy_f;
            bool jtrans = false;
            CUTEST_cfn(&status, &Nx_, &Nc_, Vec_x(x).data(), &dummy_f, c_.data());
            return c_.head(Nh);
        }

        template <typename BaseType>
        inline Mat_h Eval_grad_h(const BaseType &x)
        {
            // Gradient of objective function/lagrangian 0/1
            const bool grlagf = 0;
            const bool jtrans = 0;
            Mat_c grad_c;
            double dummy_lbd[Nc_];
            CUTEST_cgr(&status, &Nx_, &Nc_, Vec_x(x).data(), dummy_lbd, &grlagf, c_.data(), &jtrans, &Nc_, &Nx_, grad_c.data());
            return grad_c.topRows(Nh);
        }

        template <typename BaseType_x, typename BaseType_lbd>
        inline Mat_x Eval_hessian_h(const BaseType_x &x, const BaseType_lbd &lbd)
        {
            Mat_x hessian_h = Mat_x::Zero();
            Vec_c lbd_c = Vec_c::Zero();
            lbd_c.head(Nh) = lbd;
            CUTEST_cdhc(&status, &Nx_, &Nc_, Vec_x(x).data(), lbd_c.data(), &Nx_, hessian_h.data());
            return hessian_h;
        }

        template <typename BaseType>
        inline Vec_g Eval_g(const BaseType &x)
        {
            double dummy_f;
            bool jtrans = false;
            CUTEST_cfn(&status, &Nx_, &Nc_, Vec_x(x).data(), &dummy_f, c_.data());
            return c_.tail(Ng);
        }

        template <typename BaseType>
        inline Mat_g Eval_grad_g(const BaseType &x)
        {
            Mat_c grad_c;
            // Gradient of objective function/lagrangian 0/1
            const bool grlagf = 0;
            const bool jtrans = 0;
            double dummy_lbd[Nc_];
            CUTEST_cgr(&status, &Nx_, &Nc_, Vec_x(x).data(), dummy_lbd, &grlagf, c_.data(), &jtrans, &Nc_, &Nx_, grad_c.data());
            return grad_c.bottomRows(Ng);
        }

        template <typename BaseType_x, typename BaseType_lbd>
        inline Mat_x Eval_hessian_g(const BaseType_x &x, const BaseType_lbd &lbd_g)
        {
            Mat_x hessian_g = Mat_x::Zero();
            Vec_c lbd_c = Vec_c::Zero();
            lbd_c.tail(Ng) = lbd_g;
            CUTEST_cdhc(&status, &Nx_, &Nc_, Vec_x(x).data(), lbd_c.data(), &Nx_, hessian_g.data());
            return hessian_g;
        }

        inline Vec_h Get_lbd0()
        {
            return lbd0_;
        }

        inline Vec_x Get_x0()
        {
            return x0_;
        }

        inline Vec_x Get_x_ub()
        {
            return x_ub_;
        }

        inline Vec_x Get_x_lb()
        {
            return x_lb_;
        }

        ~objective_SIF_base()
        {
            FORTRAN_close(&funit, &ierr);
        }

    protected:
        const std::string fPath_;
        const int funit = 42;

        int io_buffer;
        int iout;
        int ierr;
        int status;

        Vec_x x0_;
        Vec_x x_lb_;
        Vec_x x_ub_;

        constexpr static int Nc = Ng + Nh;
        constexpr static int Nw = Nx + Ng;
        constexpr static int Ns = Ng;
        int Nx_;
        int Nc_ = 0;
        Vec_h lbd0_;
        Vec_c c_;
        Vec_c lc_;
        Vec_c uc_;

        bool eqn_type_[Nc], eqn_linear_[Nc];
    };

    template <int Nx, int Ng, int Nh>
    struct objective_SIF : public objective_SIF_base<Nx, Ng, Nh, objective>
    {
        using objective_SIF_base<Nx, Ng, Nh, objective>::objective_SIF_base;
    };

    template <int Nx, int Ng, int Nh>
    struct objective_SIF_memoized : public objective_SIF_base<Nx, Ng, Nh, objective_memoized>
    {
        using objective_SIF_base<Nx, Ng, Nh, objective_memoized>::objective_SIF_base;
    };

    template <int Nx, int Ng, int Nh>
    struct objective_SIF_journalist : public objective_SIF_base<Nx, Ng, Nh, objective_journalist>
    {
        template <typename Observer>
        objective_SIF_journalist(const std::string& fPath, Observer& Obs) : objective_SIF_base<Nx, Ng, Nh, objective_journalist>(fPath, Obs){}
    };
}
#endif