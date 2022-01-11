#ifndef FIPOPT_IC_SOLVER_HPP
#define FIPOPT_IC_SOLVER_HPP
#include <Common/EigenDataTypes.hpp>
#include <Eigen/Eigenvalues>
#include <Dense/Algorithms/Inertia_Correction/IC_Param.hpp>
#include <Dense/Algorithms/Inertia_Correction/IC_Status.hpp>
#include <Dense/KKT_System/KKT_System_Dense.hpp>
namespace FIPOPT::Dense
{

    template <int Nx, int Nh, typename T>
    inline bool Eval_Inertia_Condition(const MatrixBase<T> &KKT_mat, const double &zero_tol)
    {
        using Vec_A = Eigen::Matrix<double, Nx + Nh, 1>;
        Vec_A eigvals = KKT_mat.eigenvalues().real();
        return (((eigvals.array() > zero_tol).count() == Nx) && ((eigvals.array() < 0).count() == Nh)); // count() == Nx) && ((KKT_eigvals.real().array() < 0).count() == Nh);
    }

    template <typename Vec_A>
    inline void Update_deltas(const MatrixBase<Vec_A> &KKT_eigvals,
                              double &delta_c,
                              double &delta_w,
                              const double &delta_w_last,
                              const double &mu,
                              const IC_param &P)
    {
        delta_c = ((KKT_eigvals.array().abs() < P.zero_tol).any()) ? P.delta_c_bar * pow(mu, P.kappa_c) : 0;
        delta_w = (delta_w_last == 0) ? P.delta_w_0 : std::max(P.delta_w_min, P.kappa_w_n * delta_w_last);
    }

    template <template <class> typename LinSolver>
    struct IC_solver
    {
        template <typename Mat_A, typename Vec_A, typename Vec_x, typename Vec_cE>
        static IC_status Solve(
            const MatrixBase<Mat_A> &KKT_mat,
            MatrixBase<Vec_A> &KKT_vec,
            MatrixBase<Vec_x> &d_x,
            MatrixBase<Vec_cE> &d_lbd,
            double& delta_w_last,
            const double &mu,
            const std::string& fPath,
            const IC_param &P = IC_param())
        {
            constexpr static int N_A = Vec_A::RowsAtCompileTime;
            constexpr static int Nx = Vec_x::RowsAtCompileTime;
            constexpr static int Nh = Vec_cE::RowsAtCompileTime;
            std::ofstream f_deltas(fPath + "delta.csv", std::ios_base::app);
            LinSolver<Mat_A> KKT_solver(KKT_mat);
            double delta_c = 0;
            double delta_w = 0;
            Vec_A eigval = KKT_mat.eigenvalues().real();
            Mat_A delta_mat = Mat_A::Zero();

            while(delta_w < P.delta_w_max)
            {
                if (Eval_Inertia_Condition<Nx, Nh>(KKT_mat + delta_mat, P.zero_tol) && Solve_KKT_System(KKT_vec, d_x, d_lbd, KKT_solver))
                {
                    delta_w_last = (delta_w == 0) ? delta_w_last : delta_w;
                    f_deltas << delta_w << ", " << delta_c << '\n';
                    return IC_ACCEPTED;
                }
                Vec_A sol, KKT_eigvals;
                KKT_eigvals = (KKT_mat + delta_mat).eigenvalues().real();
                Update_deltas(KKT_eigvals, delta_c, delta_w, delta_w_last, mu, P);
                delta_mat.topLeftCorner(Nx, Nx).diagonal().array() = delta_w;
                delta_mat.bottomRightCorner(Nh, Nh).diagonal().array() = -delta_c;
                KKT_solver.compute(KKT_mat + delta_mat);
                delta_w_last = (delta_w_last == 0) ? P.kappa_w_p_bar * delta_w : P.kappa_w_p * delta_w;

            }
            return IC_INFEASIBLE;
        }
    };


}
#endif