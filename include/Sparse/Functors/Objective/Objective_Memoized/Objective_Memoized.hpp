#ifndef FIPOPT_OBJECTIVE_MEMOIZED_SPARSE_HPP
#define FIPOPT_OBJECTIVE_MEMOIZED_SPARSE_HPP
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Objective/Objective.hpp>
#include <Common/Memoizer/Memoizer.hpp>

namespace FIPOPT::Sparse
{
    template <int buffer_size>
    struct NLP_memoizer
    {
        // Dense types


        memoizer<dVec, Val, buffer_size> Eval;
        memoizer<dVec, dVec, buffer_size> Eval_grad;
        memoizer<dVec, spMat, buffer_size> Eval_hessian_f;
        memoizer<dVec, dVec, buffer_size> Eval_h;
        memoizer<dVec, spMat, buffer_size> Eval_grad_h;
        memoizer<dVec, spMat, buffer_size> Eval_hessian_h;
        memoizer<dVec, dVec, buffer_size> Eval_g;
        memoizer<dVec, spMat, buffer_size> Eval_grad_g;
        memoizer<dVec, spMat, buffer_size> Eval_hessian_g;

        NLP_memoizer() : Eval("Eval"),
                                     Eval_grad("Eval_grad"),
                                     Eval_hessian_f("Eval_hessian_f"),
                                     Eval_h("Eval_h"),
                                     Eval_hessian_h("Eval_hessian_h"),
                                     Eval_grad_h("Eval_grad_h"),
                                     Eval_g("Eval_g"),
                                     Eval_grad_g("Eval_grad_g"),
                                     Eval_hessian_g("Eval_hessian_g") {}
    };

    template <typename Derived>
    struct objective_memoized : public objective<objective_memoized<Derived>>
    {
        // Dense types


    private:
        static const int buffer_size_ = 4;
        NLP_memoizer<buffer_size_> mem;

    public:
        inline Val operator()(const MatrixBase<dVec> &x)
        {
            Val res;
            if (mem.Eval.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->operator()(x);
                mem.Eval.Set_Data(x, res);
            }
            return res;
        }

        inline dVec Eval_grad(const MatrixBase<dVec> &x)
        {
            dVec res;
            if (mem.Eval_grad.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad(x);
                mem.Eval_grad.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_hessian_f(const MatrixBase<dVec> &x)
        {
            spMat res;
            if (mem.Eval_hessian_f.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian_f(x);
                mem.Eval_hessian_f.Set_Data(x, res);
            }
            return res;
        }

        inline dVec Eval_h(const MatrixBase<dVec> &x)
        {
            dVec res;
            if (mem.Eval_h.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_h(x);
                mem.Eval_h.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_grad_h(const MatrixBase<dVec> &x)
        {
            spMat res;
            if (mem.Eval_grad_h.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad_h(x);
                mem.Eval_grad_h.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_hessian_h(const MatrixBase<dVec> &x, const MatrixBase<dVec> &lbd)
        {
            spMat res;
            if (mem.Eval_hessian_h.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian_h(x, lbd);
                mem.Eval_hessian_h.Set_Data(x, res);
            }
            return res;
        }

        inline dVec Eval_g(const MatrixBase<dVec> &x)
        {
            dVec res;
            if (mem.Eval_g.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_g(x);
                mem.Eval_g.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_grad_g(const MatrixBase<dVec> &x)
        {
            spMat res;
            if (mem.Eval_grad_g.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad_g(x);
                mem.Eval_grad_g.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_hessian_g(const MatrixBase<dVec> &x, const MatrixBase<dVec> &lbd_g)
        {
            spMat res;
            if (mem.Eval_hessian_g.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian_g(x, lbd_g);
                mem.Eval_hessian_g.Set_Data(x, res);
            }
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