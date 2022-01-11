#ifndef OBJECTIVE_QP_HPP
#define OBJECTIVE_QP_HPP
#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Objective/Objective_Memoized/Objective_Memoized.hpp>
#include <Dense/Functors/Objective/Objective_Messenger/Objective_Messenger.hpp>
#include <Dense/Functors/Objective/Objective_Journalist/Objective_Journalist.hpp>
#include <fstream>

namespace FIPOPT::Dense
{

    template <int Nx, int Ng, int Nh, template <class, int, int, int> typename Base = objective>
    struct objective_QP_base : public Base<objective_QP_base<Nx, Ng, Nh, Base>, Nx, Ng, Nh>
    {
        using Mat_x = Eigen::Matrix<double, Nx, Nx>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Mat_h = Eigen::Matrix<double, Nh, Nx>;
        using Vec_h = Eigen::Matrix<double, Nh, 1>;
        using Mat_g = Eigen::Matrix<double, Ng, Nx>;
        using Vec_g = Eigen::Matrix<double, Ng, 1>;
        using base_t = Base<objective_QP_base<Nx, Ng, Nh, Base>, Nx, Ng, Nh>;

    protected:
        const Mat_x Q_;
        const Vec_x c_;
        const Mat_h A_;
        const Vec_h b_;
        const Mat_g D_;
        const Vec_g e_;
        const Vec_x x_lb_;
        const Vec_x x_ub_;

    public:
        template <typename... Args>
        objective_QP_base(const Mat_x &Q,
                                const Vec_x &c,
                                const Mat_h &A,
                                const Vec_h &b,
                                const Mat_g &D,
                                const Vec_g &e,
                                const Vec_x &x_lb,
                                const Vec_x &x_ub,
                                Args... args) : Q_{Q}, c_{c}, A_{A}, b_{b}, D_{D}, e_{e}, x_lb_{x_lb},
                                                x_ub_{x_ub}, base_t(args...) {}
        template <typename... Args>
        objective_QP_base(const Mat_x &Q,
                                const Vec_x &c,
                                const Vec_x &x_lb,
                                const Vec_x &x_ub,
                                Args... args) : Q_(Q), c_(c), x_lb_(x_lb),
                                                x_ub_(x_ub), base_t(args...) {}

        template <typename BaseType>
        inline Val operator()(const BaseType &x)
        {
            return 0.5 * x.transpose() * Q_ * x + c_.transpose() * x;
        }

        template <typename BaseType>
        inline Vec_x Eval_grad(const BaseType &x)
        {
            return Q_.transpose() * x + c_;
        }

        template <typename BaseType>
        inline Mat_x Eval_hessian_f(const BaseType &x)
        {
            return Q_;
        }

        template <typename BaseType>
        inline Vec_h Eval_h(const BaseType &x)
        {
            return A_ * x - b_;
        }

        template <typename BaseType>
        inline Mat_h Eval_grad_h(const BaseType &x)
        {
            return A_;
        }

        template <typename BaseType_x, typename BaseType_lbd>
        inline Mat_x Eval_hessian_h(const BaseType_x &x, const BaseType_lbd &lbd)
        {
            return Mat_x::Zero();
        }

        template <typename BaseType>
        inline Vec_g Eval_g(const BaseType &x)
        {
            return D_ * x - e_;
        }

        template <typename BaseType>
        inline Mat_g Eval_grad_g(const BaseType &x)
        {
            return D_;
        }

        template <typename BaseType_x, typename BaseType_lbd>
        inline Mat_x Eval_hessian_g(const BaseType_x &x, const BaseType_lbd &lbd_g)
        {
            return Mat_x::Zero();
        }

        inline Vec_x Get_x_lb()
        {
            return x_lb_;
        }

        inline Vec_x Get_x_ub()
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
    struct objective_QP : public objective_QP_base<Nx, Ng, Nh, objective>
    {
        using objective_QP_base<Nx, Ng, Nh, objective>::objective_QP_base;
    };

    template <int Nx, int Ng, int Nh>
    struct objective_QP_memoized : public objective_QP_base<Nx, Ng, Nh, objective_memoized>
    {
        using objective_QP_base<Nx, Ng, Nh, objective_memoized>::objective_QP_base;
    };

    template <int Nx, int Ng, int Nh>
    struct objective_QP_journalist : public objective_QP_base<Nx, Ng, Nh, objective_journalist>
    {
        template <typename Observer>
        objective_QP_journalist(const std::string& fPath, Observer& Obs) : objective_QP_base<Nx, Ng, Nh, objective_journalist>(fPath, Obs){}
    };


}

#endif