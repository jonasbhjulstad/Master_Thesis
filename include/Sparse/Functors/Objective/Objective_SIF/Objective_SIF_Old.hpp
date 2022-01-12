#ifndef FIPOPT_OBJECTIVE_SIF_SPARSE_HPP
#define FIPOPT_OBJECTIVE_SIF_SPARSE_HPP
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <Common/Cutest.hpp>
#include "SIF_Parameters.hpp"
#include <Functors/Objective/Objective.hpp>
#include <Common/EigenDataTypes.hpp>
#include <Initial_Multipliers/Initial_Multipliers.hpp>

namespace FIPOPT
{
    template <template <class, int, int> typename Base>
    struct objective_SIF_base : public Base<objective_SIF_base<Base>, Nx, Ng + Nh>
    {

    public:
        objective_SIF_base(
            const std::string &fPath_) : fPath(fPath_)
        {
            Allocate();
        };

        template <typename BaseType>
        inline spVal operator()(const BaseType &x)
        {
            spVal f;
            CUTEST_ufn(&status, &Nx_, Vec_w(w).data(), f(0,0));
            return f;
        }

        template <typename BaseType>
        inline dVec Eval_grad(const BaseType &x)
        {
            CUTEST_ugr(&status, &Nx_, Vec_w(w).data(), grad_f_.data());
            return grad_f_;
        }

        template <typename BaseType>
        inline dVec Eval_grad(const BaseType &x)
        {
            return Eval_grad_dense(w).sparseView();
        }

        template <typename BaseType>
        inline spMat Eval_hessian(const BaseType &x)
        {
            CUTEST_ush(&status, &Nx_, Vec_w(w).data(), &nnzh_, &lh_, &h_[0], &irnh_[0], &icnh_[0]);
            hessian_triplets.clear();
            for (int i = 0; i < nnzh_; i++)
            {
                hessian_triplets.push_back(Triplet(irnh_[i] - 1, icnh_[i] - 1, h_[i]));
            }
            hessian_.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
            return hessian_;
        }

        template <typename BaseType>
        inline dVec Eval_c(const BaseType &x)
        {
            double dummy_f;
            CUTEST_cfn(&status, &Nx_, &Nc_, x.data(), &dummy_f, c_.data());
            return c_.sparseView();
        }

        template <typename BaseType>
        inline spMat Eval_grad_c(const BaseType &x)
        {
            // Gradient of objective function/lagrangian 0/1
            const bool grlagf = 0;

            int lj1, lj2;

            CUTEST_cdimsj(&status, &nnzj_);

            grad_c_triplets_.clear();

            bool grad = true;
            int length_J;
            CUTEST_ccfsg(&status, &Nx_, &Nc_, Vec_w(w).data(), c_.data(), &nnzj_, &length_J, J_val_, J_var_, J_fun_, &grad);
            for (int i = 0; i < nnzj_; i++)
            {
                grad_c_triplets_.push_back(Triplet(J_var_[i] - 1, J_fun_[i] - 1, J_val_[i]));
            }
            // Set slack-jacobian identity
            for (int i = 0; i < Ns; i++)
            {
                grad_c_triplets_.push_back(Triplet(Nx + i, i, 1.));
            }

            grad_c_.setFromTriplets(grad_c_triplets_.begin(), grad_c_triplets_.end());

            return grad_c_;
        }

        inline Vec_c Get_lbd0()
        {
            return lbd0_;
        }

        inline Vec_s Get_s0()
        {
            return s0_;
        }

        inline dVec Get_x0()
        {
            return x0_;
        }
        
        inline Vec_w Get_w0()
        {
            return w0_;
        }

        ~objective_SIF_base_sparse()
        {
            FORTRAN_close(&funit, &ierr);
            FREE(h_);
            FREE(icnh_);
            FREE(irnh_);

            FREE(J_val_);
            FREE(J_var_);
            FREE(J_fun_);

        }

    protected:
        const std::string fPath;
        int funit;

        int io_buffer;
        int iout;
        int ierr;
        int status;

        int lh_;
        double *h_;
        int *irnh_;
        int *icnh_;

        std::vector<Triplet> hessian_triplets;
        dVec x0_;
        dVec x_lb_;
        dVec x_ub_;
        Vec_w grad_f_;
        spMat hessian_;

        constexpr static int Nc = Ng + Nh;
        constexpr static int Nw = Nx + Ng;
        constexpr static int Ns = Ng;
        int Nx_;
        int Nc_;
        Vec_c lbd0_;
        Vec_s s0_;
        Vec_w w0_;
        Vec_c lc_;
        Vec_c uc_;
        Vec_c c_;
        double *J_val_;
        int *J_fun_, *J_var_;
        std::vector<Triplet> grad_c_triplets_;
        spMat grad_c_;

        bool eqn_type_[Nc], eqn_linear_[Nc];

    private:
        void Allocate()
        {
            MALLOC(h_, 2 * nnzh_, double);
            MALLOC(icnh_, 2 * nnzh_, int);
            MALLOC(irnh_, 2 * nnzh_, int);
            hessian_triplets.reserve(2 * nnzh_);

            hessian_.resize(Nw, Nw);
            hessian_.reserve(2 * nnzh_);

            c_.resize(Nc);
            grad_c_.resize(Nc, Nw);
            grad_c_.reserve(2 * nnzj_);
            grad_c_triplets_.reserve(2 * nnzj_);

            MALLOC(J_val_, 2 * nnzj_, double);
            MALLOC(J_var_, 2 * nnzj_, int);
            MALLOC(J_fun_, 2 * nnzj_, int);
        }

        void Read_Parameters()
        {
            int dummy;
            FORTRAN_open(&funit, &fPath[0], &ierr);

            if constexpr (Nc > 0)
            {
                int e_order, l_order, v_order;
                // Equalities should appear before inequalities:
                e_order = 1;
                // No imposed order on linear/nonlinear constraints
                l_order = 0;
                v_order = 0;

                CUTEST_cdimen(&status, &funit, &Nx_, &Nc_);
                CUTEST_csetup(&status, &funit, &iout, &io_buffer,
                              &Nx_, &Nc_, x0_.data(), x_lb_.data(), x_ub_.data(),
                              lbd0_.data(), lc_.data(), uc_.data(),
                              eqn_type_, eqn_linear_, &e_order, &l_order, &v_order);
                CUTEST_cdimsj(&status, &nnzj_);
                CUTEST_cdimsh(&status, &nnzh_);

                // Compute initial slack-variables:
                Vec_c c0;
                double f_dummy;
                CUTEST_cfn(&status, &Nx_, &Nc_, x0_.data(), &f_dummy, c0.data());

                s0_ = c0.tail(Ng);
                
                w0_ << x0_, s0_;
            }
            else
            {
                CUTEST_udimen(&status, &funit, &Nx_);
                CUTEST_usetup(&status, &funit, &iout, &io_buffer, &Nx_, x0_.data(), x_lb_.data(), x_ub_.data());
            }
        }
    };

}
#endif