#ifndef FIPOPT_CORRECTOR_HPP
#define FIPOPT_CORRECTOR_HPP
#include <Sparse/Functors/Corrector/Corrector_Status.hpp>

namespace FIPOPT::Sparse
{
    template <typename Derived>
    struct corrector
    {
        template <typename dVec, typename dVec, typename LinSolver>
        inline corrector_status Eval_System(MatrixBase<dVec> &d_x,
                                            MatrixBase<dVec> &d_lbd,
                                            double &alpha,
                                            SolverBase<LinSolver> &KKT_solver)
        {
            return static_cast<Derived *>(this)->Eval_System(d_x, d_lbd, alpha, KKT_solver);
        }

        template <typename dVec, typename dVec, typename dVec>
        inline void Update_State(const dVec &x, const dVec &lbd, const dVec &z)
        {
            static_cast<Derived *>(this)->Update_State(x, lbd, z);
        }
    };
}

#endif