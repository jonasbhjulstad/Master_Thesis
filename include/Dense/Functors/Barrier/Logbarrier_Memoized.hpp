#ifndef FIPOPT_LOGBARRIER_MEMOIZED_DENSE_HPP
#define FIPOPT_LOGBARRIER_MEMOIZED_DENSE_HPP
#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Barrier/Barrier_Memoized/Barrier_Memoized.hpp>
#include <Dense/Functors/Barrier/Logbarrier.hpp>

namespace FIPOPT::Dense
{

    template <typename Derived, int Nx, int Ng, int Nh>
    struct logbarrier_memoized : public logbarrier_base<Derived, Nx, Ng, Nh, barrier_memoized>
    {
        using logbarrier_base<Derived, Nx, Ng, Nh, barrier_memoized>::logbarrier_base;
    };

    template <typename Derived, int Nx, int Ng, int Nh>
    logbarrier_memoized(objective<Derived, Nx, Ng, Nh>& f, const double& mu) -> logbarrier_memoized<Derived, Nx, Ng, Nh>;
}

#endif