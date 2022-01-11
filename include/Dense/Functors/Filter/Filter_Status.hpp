#ifndef FILTER_STATUS_HPP
#define FILTER_STATUS_HPP
namespace FIPOPT::Dense
{

    enum filter_status
    {
        FILTER_INFEASIBLE_STEP_SIZE,
        FILTER_REJECTED,
        FILTER_ACCEPTED
    };

}

#endif