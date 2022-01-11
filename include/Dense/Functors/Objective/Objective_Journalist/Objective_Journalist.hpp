#ifndef FIPOPT_OBJECTIVE_JOURNALIST_DENSE_HPP
#define FIPOPT_OBJECTIVE_JOURNALIST_DENSE_HPP
#include <Dense/Functors/Objective/Objective_Journalist/Observer_Journalist.hpp>
#include <Dense/Functors/Objective/Objective_Messenger/Objective_Messenger.hpp>
namespace FIPOPT::Dense
{
    template <typename Derived, int Nx, int Ng, int Nh>
    struct objective_journalist: public objective_messenger<Derived, Nx, Ng, Nh, observer_journalist<Nx, Ng, Nh>>
    {
        using objective_messenger<Derived, Nx, Ng, Nh, observer_journalist<Nx, Ng, Nh>>::objective_messenger;
    };
}
#endif