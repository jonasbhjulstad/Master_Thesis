#ifndef FIPOPT_CORRECTOR_HPP
#define FIPOPT_CORRECTOR_HPP
#include <Sparse/Functors/Corrector/Corrector_Status.hpp>

namespace FIPOPT::Sparse
{
    template <typename Derived>
    struct corrector
    {
        template <typename spVec, typename spVec, typename LinSolver>
        inline corrector_status Eval_System(MatrixBase<spVec> &d_x,
                                            MatrixBase<spVec> &d_lbd,
                                            double &alpha,
                                            SolverBase<LinSolver> &KKT_solver)
        {
            return static_cast<Derived *>(this)->Eval_System(d_x, d_lbd, alpha, KKT_solver);
        }

        template <typename spVec, typename spVec, typename spVec>
        inline void Update_State(const spVec &x, const spVec &lbd, const spVec &z)
        {
            static_cast<Derived *>(this)->Update_State(x, lbd, z);
        }
    };
}

#endif