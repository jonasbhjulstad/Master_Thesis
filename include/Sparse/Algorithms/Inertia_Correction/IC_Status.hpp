#ifndef FIPOPT_IC_STATUS_HPP
#define FIPOPT_IC_STATUS_HPP

namespace FIPOPT::Sparse
{
    enum IC_status
    {
        IC_INFEASIBLE = -1,
        IC_MAX_ITERATION_EXCEEDED,
        IC_ACCEPTED
    };
}

#endif