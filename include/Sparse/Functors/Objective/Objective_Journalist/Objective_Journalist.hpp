#ifndef FIPOPT_OBJECTIVE_JOURNALIST_SPARSE_HPP
#define FIPOPT_OBJECTIVE_JOURNALIST_SPARSE_HPP
#include <Dense/Functors/Objective/Objective_Journalist/Observer_Journalist.hpp>
#include <Dense/Functors/Objective/Objective_Messenger/Objective_Messenger.hpp>
namespace FIPOPT::Sparse
{
    template <typename Derived>
    struct objective_journalist: public objective_messenger<Derived, observer_journalist>
    {
        using objective_messenger<Derived, observer_journalist>::objective_messenger;
    };
}
#endif