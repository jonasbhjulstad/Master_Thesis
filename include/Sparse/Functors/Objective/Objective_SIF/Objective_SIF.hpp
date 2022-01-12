#ifndef FIPOPT_OBJECTIVE_SIF_SPARSE_HPP
#define FIPOPT_OBJECTIVE_SIF_SPARSE_HPP
#include <iostream>
#include <string>
#include <vector>
#include <Common/Cutest.hpp>
// #include <Sparse/Functors/Objective/Objective_Memoized/Objective_Memoized.hpp>
#include <Sparse/Functors/Objective/Objective.hpp>
// #include <Sparse/Functors/Objective/objective_messenger/objective_messenger.hpp>
#include <Common/EigenDataTypes.hpp>

namespace FIPOPT::Sparse
{
    template <template <class> typename Base = objective>
    struct objective_SIF_base : public Base<objective_SIF_base<Base>>
    {
        using base_t = Base<objective_SIF_base<Base>>;

    private:
        void Setup()
        {
            int dummy;
            FORTRAN_open(&funit, &fPath_[0], &ierr);
            CUTEST_cdimen(&status, &funit, &Nx_, &Nc_);

            x0_.resize(Nx_);
            x_lb_.resize(Nx_);
            x_ub_.resize(Nx_);

            if (Nc_ > 0)
            {
                int e_order, l_order, v_order;
                // Equalities should appear before inequalities:
                e_order = 1;
                // No imposed order on linear/nonlinear constraints
                l_order = 0;
                v_order = 0;

                CUTEST_csetup(&status, &funit, &iout, &io_buffer,
                              &Nx_, &Nc_, x0_.data(), x_lb_.data(), x_ub_.data(),
                              c_.data(), lc_.data(), uc_.data(),
                              eqn_type_, eqn_linear_, &e_order, &l_order, &v_order);

                CUTEST_cdimsj(&status, &nnzj_);
                CUTEST_cdimsh(&status, &nnzh_);
            }
            else
            {
                Nh_ = 0;
                Ng_ = 0;
                CUTEST_usetup(&status, &funit, &iout, &io_buffer, &Nx_, x0_.data(), x_lb_.data(), x_ub_.data());
            }
        }

    public:
        objective_SIF_base(const std::string &fPath) : fPath_(fPath)
        {
            Setup();
            base_t::Initialize(Nx_, Ng_, Nh_);
            Allocate();
        }
        template <typename Observer>
        objective_SIF_base(const std::string &fPath, Observer &obs) : fPath_(fPath), base_t(obs)
        {
            Setup();
            base_t::Initialize(Nx_, Ng_, Nh_);
            Allocate();
        }

        template <typename T>
        inline spVal operator()(const MatrixBase<T> &x)
        {
            Val f;
            dVec dense_x = dVec(x);
            CUTEST_ufn(&status, &Nx_, dense_x.data(), f.data());
            return f.sparseView();
        }

        template <typename T>
        inline dVec Eval_grad(const MatrixBase<T> &x)
        {
            dVec dense_x = dVec(x);
            CUTEST_ugr(&status, &Nx_, dense_x.data(), grad_f_.data());
            return grad_f_.sparseView();
        }

        template <typename T>
        inline spMat Eval_hessian_f(const MatrixBase<T> &x)
        {
            dVec dense_x = dVec(x);
            CUTEST_ush(&status, &Nx_, dense_x.data(), &nnzh_, lh_, h_vals_, irnh_, icnh_);
            hessian_triplets_.clear();
            for (int i = 0; i < nnzh_; i++)
            {
                hessian_triplets_.push_back(Triplet(irnh_[i] - 1, icnh_[i] - 1, h_vals_[i]));
            }
            hessian_.setFromTriplets(hessian_triplets_.begin(), hessian_triplets_.end());
            return hessian_;
        }

        template <typename T>
        inline dVec Eval_h(const MatrixBase<T> &x)
        {
            return Eval_c(x).topRows(Nh_);
        }

        template <typename T>
        inline dVec Eval_grad_h(const MatrixBase<T> &x)
        {
            return Eval_grad_c(x).topRows(Nh_);
        }

        template <typename T0, typename T1>
        inline spMat Eval_hessian_h(const MatrixBase<T0> &x, const MatrixBase<T1> &lbd_h)
        {
            dVec lbd_c(Nc_);
            lbd_c.head(Nh_) = dVec(lbd_h);
            return Eval_hessian_c(x, lbd_c);
        }

        template <typename T>
        inline dVec Eval_g(const MatrixBase<T> &x)
        {
            return Eval_c(x).bottomRows(Ng_);
        }

        template <typename T>
        inline spMat Eval_grad_g(const MatrixBase<T> &x)
        {
            return Eval_grad_c(x).bottomRows(Ng_);
        }

        template <typename T0, typename T1>
        inline spMat Eval_hessian_g(const MatrixBase<T0> &x, const MatrixBase<T1> &lbd_g)
        {
            dVec lbd_c(Nc_);
            lbd_c.tail(Ng_) = dVec(lbd_g);
            return Eval_hessian_c(x, lbd_c);
        }

        inline dVec Get_lbd0()
        {
            return lbd0_;
        }

        inline dVec Get_x0()
        {
            return x0_;
        }

        inline dVec Get_x_ub()
        {
            return x_ub_;
        }

        inline dVec Get_x_lb()
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

        dVec x0_;
        dVec x_lb_;
        dVec x_ub_;
        dVec grad_f_;

        dVec lbd0_;
        dVec c_;
        dVec lc_;
        dVec uc_;
        spMat grad_c_;
        spMat hessian_c_;

        bool *eqn_type_;
        bool *eqn_linear_;

        int* lh_;
        double *h_;
        int *irnh_;
        int *icnh_;
        int *irnz_;
        int *icnz_;
        int nnzh_, nnzj_;
        int Nx_, Ng_, Nh_, Nc_;
        double *h_vals_;
        double *c_vals_;
        spMat hessian_;

        std::vector<Triplet> hessian_triplets_;
        std::vector<Triplet> grad_c_triplets_;
        std::vector<Triplet> hessian_c_triplets_;

    private:
        void Allocate()
        {
            MALLOC(h_vals_, 2 * nnzh_, double);
            MALLOC(icnh_, 2 * nnzh_, int);
            MALLOC(irnh_, 2 * nnzh_, int);

            MALLOC(lh_, Nx_, int);

            MALLOC(eqn_type_, Nc_, bool);
            MALLOC(eqn_linear_, Nc_, bool);

            MALLOC(c_vals_, 2 * nnzj_, double);
            MALLOC(icnz_, 2 * nnzj_, int);
            MALLOC(irnz_, 2 * nnzj_, int);

            hessian_triplets_.reserve(2 * nnzh_);

            hessian_.resize(Nx_, Nx_);
            hessian_.reserve(2 * nnzh_);

            c_.resize(Nc_);
            grad_c_.resize(Nc_ + 2*Nx_, Nx_);
            grad_c_.reserve(2 * nnzj_);
            grad_c_triplets_.reserve(2 * nnzj_);
            hessian_c_.resize(Nx_, Nx_);
            hessian_c_triplets_.reserve(2 * nnzj_);
            grad_f_.resize(Nx_);

        }

        template <typename T>
        inline dVec Eval_c(const MatrixBase<T> &x)
        {
            double dummy_f;
            bool jtrans = false;
            CUTEST_cfn(&status, &Nx_, &Nc_, dVec(x).data(), &dummy_f, c_.data());
            return c_.sparseView();
        }

        template <typename T>
        inline spMat Eval_grad_c(const MatrixBase<T> &x)
        {
            // Gradient of objective function/lagrangian 0/1
            const bool grlagf = 0;
            const bool jtrans = 0;
            double dummy_lbd[Nc_];
            int lh[Nc_];
            CUTEST_csgr(&status, &Nx_, &Nc_, dVec(x).data(), dummy_lbd, &grlagf, &nnzj_, lh, c_vals_, irnz_, icnz_);
            for (int i = 0; i < nnzh_; i++)
            {
                grad_c_triplets_.push_back(Triplet(irnz_[i] - 1, icnz_[i] - 1, c_vals_[i]));
            }

            grad_c_.setFromTriplets(hessian_triplets_.begin(), hessian_triplets_.end());
            return grad_c_;
        }

        template <typename T0, typename T1>
        inline spMat Eval_hessian_c(const MatrixBase<T0> &x, const MatrixBase<T1> &lbd)
        {
            int lc_[Nc_];
            double c_vals[2 * nnzj_];
            CUTEST_cshc(&status, &Nx_, &Nc_, dVec(x).data(), dVec(lbd).data(), &nnzh_, lc_, c_vals, irnz_, icnz_);

            hessian_c_triplets_.clear();
            for (int i = 0; i < nnzh_; i++)
            {
                grad_c_triplets_.push_back(Triplet(irnz_[i] - 1, icnz_[i] - 1, c_vals[i]));
            }
            hessian_c_.setFromTriplets(hessian_c_triplets_.begin(), hessian_c_triplets_.end());

            return hessian_c_;
        }
    };

    struct objective_SIF : public objective_SIF_base<objective>
    {
        using objective_SIF_base<objective>::objective_SIF_base;
    };

    // struct objective_SIF_memoized : public objective_SIF_base<objective_memoized>
    // {
    //     using objective_SIF_base<objective_memoized>::objective_SIF_base;
    // };

    // struct objective_SIF_journalist : public objective_SIF_base<journalist_callback>
    // {
    //     using objective_SIF_base<journalist_callback>::objective_SIF_base;
    // };
}
#endif