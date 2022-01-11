#ifndef FIPOPT_objective_messenger_DENSE_HPP
#define FIPOPT_objective_messenger_DENSE_HPP
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Objective/Objective_Memoized/Objective_Memoized.hpp>
#include <Dense/Functors/Objective/Objective_Observer/Objective_Observer.hpp>
#include <vector>
namespace FIPOPT::Dense
{
    template <typename Derived, int Nx, int Ng, int Nh, typename Derived_O>
    struct objective_messenger : public objective_memoized<objective_messenger<Derived, Nx, Ng, Nh, Derived_O>, Nx, Ng, Nh>
    {
        using Observer = objective_observer<Derived_O, Nx, Ng, Nh>;
        // Dense types
        using Mat_x = Eigen::Matrix<double, Nx, Nx>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Mat_h = Eigen::Matrix<double, Nh, Nx>;
        using Vec_h = Eigen::Matrix<double, Nh, 1>;
        using Mat_g = Eigen::Matrix<double, Ng, Nx>;
        using Vec_g = Eigen::Matrix<double, Ng, 1>;

        Observer &observer_;

        objective_messenger(Observer &obs) : observer_(obs) {}

        inline Val operator()(const MatrixBase<Vec_x> &x)
        {
            Val res;
            res = static_cast<Derived *>(this)->operator()(x);
            observer_.Eval_f(x, res);
            return res;
        }

        inline Vec_x Eval_grad(const MatrixBase<Vec_x> &x)
        {
            Vec_x res;
            res = static_cast<Derived *>(this)->Eval_grad(x);
            observer_.Eval_grad(x, res);
            return res;
        }


        inline Mat_x Eval_hessian_f(const MatrixBase<Vec_x> &x)
        {
            Mat_x res;
            res = static_cast<Derived *>(this)->Eval_hessian_f(x);
            observer_.Eval_hessian_f(x, res);
            return res;
        }


        inline Vec_h Eval_h(const MatrixBase<Vec_x> &x)
        {
            Vec_h res = static_cast<Derived *>(this)->Eval_h(x);
            observer_.Eval_h(x, res);
            return res;
        }


        inline Mat_h Eval_grad_h(const MatrixBase<Vec_x> &x)
        {
            Mat_h res = static_cast<Derived *>(this)->Eval_grad_h(x);
            observer_.Eval_grad_h(x, res);
            return res;
        }

        inline Mat_x Eval_hessian_h(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_h> &lbd)
        {
            Mat_x res = static_cast<Derived *>(this)->Eval_hessian_h(x, lbd);
            observer_.Eval_hessian_h(x, lbd, res);
            return res;
        }

        inline Vec_g Eval_g(const MatrixBase<Vec_x> &x)
        {
            Vec_g res = static_cast<Derived *>(this)->Eval_g(x);
            observer_.Eval_g(x, res);
            return res;
        }

        inline Mat_g Eval_grad_g(const MatrixBase<Vec_x> &x)
        {
            Mat_g res = static_cast<Derived *>(this)->Eval_grad_g(x);
            observer_.Eval_grad_g(x, res);
            return res;
        }


        inline Mat_x Eval_hessian_g(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_g> &lbd_g)
        {
            Mat_x res = static_cast<Derived *>(this)->Eval_hessian_g(x, lbd_g);
            observer_.Eval_hessian_g(x, lbd_g, res);
            return res;
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