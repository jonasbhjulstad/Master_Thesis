#ifndef FIPOPT_BS_STATUS_HPP
#define FIPOPT_BS_STATUS_HPP
namespace FIPOPT::Sparse
{

    enum BS_status
    {
        BS_INFEASIBLE,
        BS_ACCEPTED,
        BS_MAX_ITERATION_EXCEEDED
    };
}

#endif