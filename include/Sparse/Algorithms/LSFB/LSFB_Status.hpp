#ifndef FIPOPT_LSFB_STATUS_HPP
#define FIPOPT_LSFB_STATUS_HPP
namespace FIPOPT::Sparse
{
    enum LSFB_status
    {
        LSFB_INFEASIBLE,
        LSFB_ACCEPTED,
        LSFB_MAX_ITERATION_EXCEEDED
    };
}
#endif