#ifndef FIPOPT_OBJECTIVE_MEMOIZED_DENSE_HPP
#define FIPOPT_OBJECTIVE_MEMOIZED_DENSE_HPP
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Objective/Objective.hpp>
#include <Common/Memoizer/Memoizer.hpp>

namespace FIPOPT::Dense
{
    template <int Nx, int Ng, int Nh, int buffer_size>
    struct NLP_memoizer
    {
        // Dense types
        using Mat_x = Eigen::Matrix<double, Nx, Nx>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Mat_h = Eigen::Matrix<double, Nh, Nx>;
        using Vec_h = Eigen::Matrix<double, Nh, 1>;
        using Mat_g = Eigen::Matrix<double, Ng, Nx>;
        using Vec_g = Eigen::Matrix<double, Ng, 1>;

        memoizer<Vec_x, Val, buffer_size> Eval;
        memoizer<Vec_x, Vec_x, buffer_size> Eval_grad;
        memoizer<Vec_x, Mat_x, buffer_size> Eval_hessian_f;
        memoizer<Vec_x, Vec_h, buffer_size> Eval_h;
        memoizer<Vec_x, Mat_h, buffer_size> Eval_grad_h;
        memoizer<Vec_x, Mat_x, buffer_size> Eval_hessian_h;
        memoizer<Vec_x, Vec_g, buffer_size> Eval_g;
        memoizer<Vec_x, Mat_g, buffer_size> Eval_grad_g;
        memoizer<Vec_x, Mat_x, buffer_size> Eval_hessian_g;

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

    template <typename Derived, int Nx, int Ng, int Nh>
    struct objective_memoized : public objective<objective_memoized<Derived, Nx, Ng, Nh>, Nx, Ng, Nh>
    {
        // Dense types
        using Mat_x = Eigen::Matrix<double, Nx, Nx>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Mat_h = Eigen::Matrix<double, Nh, Nx>;
        using Vec_h = Eigen::Matrix<double, Nh, 1>;
        using Mat_g = Eigen::Matrix<double, Ng, Nx>;
        using Vec_g = Eigen::Matrix<double, Ng, 1>;

    private:
        static const int buffer_size_ = 4;
        NLP_memoizer<Nx, Ng, Nh, buffer_size_> mem;

    public:
        inline Val operator()(const MatrixBase<Vec_x> &x)
        {
            Val res;
            if (mem.Eval.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->operator()(x);
                mem.Eval.Set_Data(x, res);
            }
            return res;
        }

        inline Vec_x Eval_grad(const MatrixBase<Vec_x> &x)
        {
            Vec_x res;
            if (mem.Eval_grad.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad(x);
                mem.Eval_grad.Set_Data(x, res);
            }
            return res;
        }

        inline Mat_x Eval_hessian_f(const MatrixBase<Vec_x> &x)
        {
            Mat_x res;
            if (mem.Eval_hessian_f.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian_f(x);
                mem.Eval_hessian_f.Set_Data(x, res);
            }
            return res;
        }

        inline Vec_h Eval_h(const MatrixBase<Vec_x> &x)
        {
            Vec_h res;
            if (mem.Eval_h.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_h(x);
                mem.Eval_h.Set_Data(x, res);
            }
            return res;
        }

        inline Mat_h Eval_grad_h(const MatrixBase<Vec_x> &x)
        {
            Mat_h res;
            if (mem.Eval_grad_h.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad_h(x);
                mem.Eval_grad_h.Set_Data(x, res);
            }
            return res;
        }

        inline Mat_x Eval_hessian_h(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_h> &lbd)
        {
            Mat_x res;
            if (mem.Eval_hessian_h.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian_h(x, lbd);
                mem.Eval_hessian_h.Set_Data(x, res);
            }
            return res;
        }

        inline Vec_g Eval_g(const MatrixBase<Vec_x> &x)
        {
            Vec_g res;
            if (mem.Eval_g.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_g(x);
                mem.Eval_g.Set_Data(x, res);
            }
            return res;
        }

        inline Mat_g Eval_grad_g(const MatrixBase<Vec_x> &x)
        {
            Mat_g res;
            if (mem.Eval_grad_g.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad_g(x);
                mem.Eval_grad_g.Set_Data(x, res);
            }
            return res;
        }

        inline Mat_x Eval_hessian_g(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_g> &lbd_g)
        {
            Mat_x res;
            if (mem.Eval_hessian_g.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian_g(x, lbd_g);
                mem.Eval_hessian_g.Set_Data(x, res);
            }
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