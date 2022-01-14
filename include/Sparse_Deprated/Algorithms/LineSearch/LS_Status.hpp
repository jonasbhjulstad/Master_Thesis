#ifndef LS_STATUS_HPP
#define LS_STATUS_HPP

namespace FIPOPT::Sparse
{
    enum LS_status
    {
        LS_INFEASIBLE=-1,
        LS_MAX_ITERATIONS_EXCEEDED,
        LS_ACCEPTED
    };
}
#endif