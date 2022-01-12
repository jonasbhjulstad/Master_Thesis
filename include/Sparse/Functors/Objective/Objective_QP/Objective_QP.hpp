#ifndef OBJECTIVE_QP_HPP
#define OBJECTIVE_QP_HPP
#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Objective/Objective_Memoized/Objective_Memoized.hpp>
#include <Dense/Functors/Objective/Objective_Messenger/Objective_Messenger.hpp>
#include <Dense/Functors/Objective/Objective_Journalist/Objective_Journalist.hpp>
#include <fstream>

namespace FIPOPT::Sparse
{

    template <template <class, int, int, int> typename Base = objective>
    struct objective_QP_base : public Base<objective_QP_base<Base>>
    {

        using base_t = Base<objective_QP_base<Base>>;

    protected:
        const spMat Q_;
        const dVec c_;
        const spMat A_;
        const dVec b_;
        const spMat D_;
        const dVec e_;
        const dVec x_lb_;
        const dVec x_ub_;

    public:
        template <typename... Args>
        objective_QP_base(const spMat &Q,
                                const dVec &c,
                                const spMat &A,
                                const dVec &b,
                                const spMat &D,
                                const dVec &e,
                                const dVec &x_lb,
                                const dVec &x_ub,
                                Args... args) : Q_{Q}, c_{c}, A_{A}, b_{b}, D_{D}, e_{e}, x_lb_{x_lb},
                                                x_ub_{x_ub}, base_t(args...) {}
        template <typename... Args>
        objective_QP_base(const spMat &Q,
                                const dVec &c,
                                const dVec &x_lb,
                                const dVec &x_ub,
                                Args... args) : Q_(Q), c_(c), x_lb_(x_lb),
                                                x_ub_(x_ub), base_t(args...) {}

        template <typename BaseType>
        inline Val operator()(const BaseType &x)
        {
            return 0.5 * x.transpose() * Q_ * x + c_.transpose() * x;
        }

        template <typename BaseType>
        inline dVec Eval_grad(const BaseType &x)
        {
            return Q_.transpose() * x + c_;
        }

        template <typename BaseType>
        inline spMat Eval_hessian_f(const BaseType &x)
        {
            return Q_;
        }

        template <typename BaseType>
        inline dVec Eval_h(const BaseType &x)
        {
            return A_ * x - b_;
        }

        template <typename BaseType>
        inline spMat Eval_grad_h(const BaseType &x)
        {
            return A_;
        }

        template <typename BaseType_x, typename BaseType_lbd>
        inline spMat Eval_hessian_h(const BaseType_x &x, const BaseType_lbd &lbd)
        {
            return spMat::Zero();
        }

        template <typename BaseType>
        inline dVec Eval_g(const BaseType &x)
        {
            return D_ * x - e_;
        }

        template <typename BaseType>
        inline spMat Eval_grad_g(const BaseType &x)
        {
            return D_;
        }

        template <typename BaseType_x, typename BaseType_lbd>
        inline spMat Eval_hessian_g(const BaseType_x &x, const BaseType_lbd &lbd_g)
        {
            return spMat::Zero();
        }

        inline dVec Get_x_lb()
        {
            return x_lb_;
        }

        inline dVec Get_x_ub()
        {
            return x_ub_;
        }

        void write_parameters()
        {
            using namespace Eigen;
            const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");

            std::ofstream file("Q.csv");
            file << Q_.format(CSVFormat);
            file.close();
            file.open("A.csv");
            file << A_.format(CSVFormat);
            file.close();
            file.open("b.csv");
            file << b_.format(CSVFormat);
            file.close();
            file.open("c.csv");
            file << c_.format(CSVFormat);
            file.close();
            file.open("D.csv");
            file << D_.format(CSVFormat);
            file.close();
            file.open("e.csv");
            file << e_.format(CSVFormat);
            file.close();
            file.open("x_lb.csv");
            file << x_lb_.format(CSVFormat);
            file.close();
            file.open("x_ub.csv");
            file << x_ub_.format(CSVFormat);
            file.close();
        }
    };

    template <int Nx, int Ng, int Nh>
    struct objective_QP : public objective_QP_base<objective>
    {
        using objective_QP_base<objective>::objective_QP_base;
    };

    template <int Nx, int Ng, int Nh>
    struct objective_QP_memoized : public objective_QP_base<objective_memoized>
    {
        using objective_QP_base<objective_memoized>::objective_QP_base;
    };

    template <int Nx, int Ng, int Nh>
    struct objective_QP_journalist : public objective_QP_base<objective_journalist>
    {
        template <typename Observer>
        objective_QP_journalist(const std::string& fPath, Observer& Obs) : objective_QP_base<objective_journalist>(fPath, Obs){}
    };


}

#endif