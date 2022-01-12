#ifndef FIPOPT_objective_messenger_SPARSE_HPP
#define FIPOPT_objective_messenger_SPARSE_HPP
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Objective/Objective_Memoized/Objective_Memoized.hpp>
#include <Dense/Functors/Objective/Objective_Observer/Objective_Observer.hpp>
#include <vector>
namespace FIPOPT::Sparse
{
    template <typename Derived, typename Derived_O>
    struct objective_messenger : public objective_memoized<objective_messenger<Derived, Derived_O>>
    {
        using Observer = objective_observer<Derived_O>;
        // Dense types


        Observer &observer_;

        objective_messenger(Observer &obs) : observer_(obs) {}

        inline Val operator()(const MatrixBase<dVec> &x)
        {
            Val res;
            res = static_cast<Derived *>(this)->operator()(x);
            observer_.Eval_f(x, res);
            return res;
        }

        inline dVec Eval_grad(const MatrixBase<dVec> &x)
        {
            dVec res;
            res = static_cast<Derived *>(this)->Eval_grad(x);
            observer_.Eval_grad(x, res);
            return res;
        }


        inline spMat Eval_hessian_f(const MatrixBase<dVec> &x)
        {
            spMat res;
            res = static_cast<Derived *>(this)->Eval_hessian_f(x);
            observer_.Eval_hessian_f(x, res);
            return res;
        }


        inline dVec Eval_h(const MatrixBase<dVec> &x)
        {
            dVec res = static_cast<Derived *>(this)->Eval_h(x);
            observer_.Eval_h(x, res);
            return res;
        }


        inline spMat Eval_grad_h(const MatrixBase<dVec> &x)
        {
            spMat res = static_cast<Derived *>(this)->Eval_grad_h(x);
            observer_.Eval_grad_h(x, res);
            return res;
        }

        inline spMat Eval_hessian_h(const MatrixBase<dVec> &x, const MatrixBase<dVec> &lbd)
        {
            spMat res = static_cast<Derived *>(this)->Eval_hessian_h(x, lbd);
            observer_.Eval_hessian_h(x, lbd, res);
            return res;
        }

        inline dVec Eval_g(const MatrixBase<dVec> &x)
        {
            dVec res = static_cast<Derived *>(this)->Eval_g(x);
            observer_.Eval_g(x, res);
            return res;
        }

        inline spMat Eval_grad_g(const MatrixBase<dVec> &x)
        {
            spMat res = static_cast<Derived *>(this)->Eval_grad_g(x);
            observer_.Eval_grad_g(x, res);
            return res;
        }


        inline spMat Eval_hessian_g(const MatrixBase<dVec> &x, const MatrixBase<dVec> &lbd_g)
        {
            spMat res = static_cast<Derived *>(this)->Eval_hessian_g(x, lbd_g);
            observer_.Eval_hessian_g(x, lbd_g, res);
            return res;
        }

        inline dVec Get_x_lb()
        {
            return static_cast<Derived *>(this)->Get_x_lb();
        }
        inline dVec Get_x_ub()
        {
            return static_cast<Derived *>(this)->Get_x_ub();
        }
    };

}
#endif