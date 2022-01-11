#ifndef FIPOPT_BARRIER_MESSENGER_DENSE_HPP
#define FIPOPT_BARRIER_MESSENGER_DENSE_HPP
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Barrier/Barrier_Memoized/Barrier_Memoized.hpp>
#include <Dense/Functors/Barrier/Barrier_Observer/Barrier_Observer.hpp>
#include <Dense/Functors/Barrier/Barrier_Journalist/Barrier_Journalist.hpp>
#include <vector>
namespace FIPOPT::Dense
{
    template <typename Derived, int Nx, int Ng, int Nh, typename Derived_O>
    struct barrier_messenger : public barrier_memoized<barrier_messenger<Derived, Nx, Ng, Nh, Derived_O>, Nx, Ng, Nh>
    {
        using Observer = barrier_observer<Derived_O, Nx, Ng, Nh>;
        // Dense types
        using Mat_x = Eigen::Matrix<double, Nx, Nx>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Mat_h = Eigen::Matrix<double, Nh, Nx>;
        using Vec_h = Eigen::Matrix<double, Nh, 1>;
        using Mat_g = Eigen::Matrix<double, Ng, Nx>;
        using Vec_g = Eigen::Matrix<double, Ng, 1>;

        Observer &observer_;

        barrier_messenger(Observer &obs, const double& mu) : observer_(obs) 
        {
        }

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
    };


}
#endif