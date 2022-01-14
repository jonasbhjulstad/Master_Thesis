#ifndef FIPOPT_OBJECTIVE_MEMOIZED_Sparse_HPP
#define FIPOPT_OBJECTIVE_MEMOIZED_Sparse_HPP
#include <Common/EigenDataTypes.hpp>
#include <Sparse/Functors/Objective/Objective.hpp>
#include <Common/Memoizer/Memoizer.hpp>

namespace FIPOPT::Sparse
{
    template <int buffer_size>
    struct NLP_memoizer_Sparse_Sparse
    {
        memoizer<spVec, spVal, buffer_size> Eval;
        memoizer<spVec, spVec, buffer_size> Eval_grad;
        memoizer<spVec, spMat, buffer_size> Eval_hessian_f;
        memoizer<spVec, spVec, buffer_size> Eval_h;
        memoizer<spVec, spMat, buffer_size> Eval_grad_h;
        memoizer<spVec, spMat, buffer_size> Eval_hessian_h;
        memoizer<spVec, spVec, buffer_size> Eval_g;
        memoizer<spVec, spMat, buffer_size> Eval_grad_g;
        memoizer<spVec, spMat, buffer_size> Eval_hessian_g;

        NLP_memoizer_Sparse_Sparse() : Eval("SS_Eval"),
                                     Eval_grad("SS_Eval_grad"),
                                     Eval_hessian_f("SS_Eval_hessian_f"),
                                     Eval_h("SS_Eval_h"),
                                     Eval_hessian_h("SS_Eval_hessian_h"),
                                     Eval_grad_h("SS_Eval_grad_h"),
                                     Eval_g("SS_Eval_g"),
                                     Eval_grad_g("SS_Eval_grad_g"),
                                     Eval_hessian_g("SS_Eval_hessian_g") {}
    };

    template <int buffer_size>
    struct NLP_memoizer_Sparse_Dense
    {

        memoizer<dVec, spVal, buffer_size> Eval;
        memoizer<dVec, spVec, buffer_size> Eval_grad;
        memoizer<dVec, spMat, buffer_size> Eval_hessian_f;
        memoizer<dVec, spVec, buffer_size> Eval_h;
        memoizer<dVec, spMat, buffer_size> Eval_grad_h;
        memoizer<dVec, spMat, buffer_size> Eval_hessian_h;
        memoizer<dVec, spVec, buffer_size> Eval_g;
        memoizer<dVec, spMat, buffer_size> Eval_grad_g;
        memoizer<dVec, spMat, buffer_size> Eval_hessian_g;


        NLP_memoizer_Sparse_Dense() : Eval("SD_Eval"),
                                      Eval_grad("SD_Eval_grad"),
                                      Eval_hessian_f("SD_Eval_hessian_f"),
                                      Eval_h("SD_Eval_h"),
                                      Eval_grad_h("SD_Eval_grad_h"),
                                      Eval_hessian_h("SD_Eval_hessian_h"),
                                      Eval_g("SD_Eval_g"),
                                      Eval_grad_g("SD_Eval_grad_g"),
                                      Eval_hessian_g("SD_Eval_hessian_g") {}
    };

    template <typename Derived>
    struct objective_memoized : public objective<objective_memoized<Derived>>
    {


    private:
        static const int buffer_size_ = 4;
        NLP_memoizer_Sparse_Sparse<buffer_size_> mem_ss;
        NLP_memoizer_Sparse_Dense<buffer_size_> mem_sd;

    public:
        inline spVal operator()(const MatrixBase<spVec> &x)
        {
            spVal res;
            if (mem_ss.Eval.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->operator()(x);
                mem_ss.Eval.Set_Data(x, res);
            }
            return res;
        }

        inline spVal operator()(const SparseMatrixBase<spVec> &x)
        {
            spVal res;
            if (mem_sd.Eval.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->operator()(x);
                mem_sd.Eval.Set_Data(x, res);
            }
            return res;
        }

        inline spVec Eval_grad(const MatrixBase<spVec> &x)
        {
            spVec res;
            if (mem_ss.Eval_grad.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad(x);
                mem_ss.Eval_grad.Set_Data(x, res);
            }
            return res;
        }

        inline spVec Eval_grad(const SparseMatrixBase<spVec> &x)
        {
            spVec res;
            if (mem_sd.Eval_grad.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad(x);
                mem_sd.Eval_grad.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_hessian_f(const MatrixBase<spVec> &x)
        {
            spMat res;
            if (mem_ss.Eval_hessian_f.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian_f(x);
                mem_ss.Eval_hessian_f.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_hessian_f(const SparseMatrixBase<spVec> &x)
        {
            spMat res;
            if (mem_sd.Eval_hessian_f.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian_f(x);
                mem_sd.Eval_hessian_f.Set_Data(x, res);
            }
            return res;
        }

        inline spVec Eval_h(const MatrixBase<spVec> &x)
        {
            spVec res;
            if (mem_ss.Eval_h.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_h(x);
                mem_ss.Eval_h.Set_Data(x, res);
            }
            return res;
        }

        inline spVec Eval_h(const SparseMatrixBase<spVec> &x)
        {
            spVec res;
            if (mem_sd.Eval_h.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_h(x);
                mem_sd.Eval_h.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_grad_h(const MatrixBase<spVec> &x)
        {
            spMat res;
            if (mem_ss.Eval_grad_h.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad_h(x);
                mem_ss.Eval_grad_h.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_grad_h(const SparseMatrixBase<spVec> &x)
        {
            spMat res;
            if (mem_sd.Eval_grad_h.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad_h(x);
                mem_sd.Eval_grad_h.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_hessian_h(const MatrixBase<spVec> &x, const MatrixBase<spVec> &lbd)
        {
            spMat res;
            if (mem_ss.Eval_hessian_h.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian_h(x, lbd);
                mem_ss.Eval_hessian_h.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_hessian_h(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVec> &lbd)
        {
            spMat res;
            if (mem_sd.Eval_hessian_h.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian_h(x, lbd);
                mem_sd.Eval_hessian_h.Set_Data(x, res);
            }
            return res;
        }

        inline spVec Eval_g(const MatrixBase<spVec> &x)
        {
            spVec res;
            if (mem_ss.Eval_g.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_g(x);
                mem_ss.Eval_g.Set_Data(x, res);
            }
            return res;
        }

        inline spVec Eval_g(const SparseMatrixBase<spVec> &x)
        {
            spVec res;
            if (mem_sd.Eval_g.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_g(x);
                mem_sd.Eval_g.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_grad_g(const MatrixBase<spVec> &x)
        {
            spMat res;
            if (mem_ss.Eval_grad_g.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad_g(x);
                mem_ss.Eval_grad_g.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_grad_g(const SparseMatrixBase<spVec> &x)
        {
            spMat res;
            if (mem_sd.Eval_grad_g.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad_g(x);
                mem_sd.Eval_grad_g.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_hessian_g(const MatrixBase<spVec> &x, const MatrixBase<spVec> &lbd_g)
        {
            spMat res;
            if (mem_ss.Eval_hessian_g.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian_g(x, lbd_g);
                mem_ss.Eval_hessian_g.Set_Data(x, res);
            }
            return res;
        }

        inline spMat Eval_hessian_g(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVec> &lbd_g)
        {
            spMat res;
            if (mem_sd.Eval_hessian_g.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian_g(x, lbd_g);
                mem_sd.Eval_hessian_g.Set_Data(x, res);
            }
            return res;
        }

        inline spVec Get_x_lb()
        {
            return static_cast<Derived *>(this)->Get_x_lb();
        }
        inline spVec Get_x_ub()
        {
            return static_cast<Derived *>(this)->Get_x_ub();
        }
    };

}
#endif