#ifndef FIPOPT_FILTER_HPP
#define FIPOPT_FILTER_HPP
#include <Sparse/Functors/Filter/Filter_Status.hpp>
#include <string>
namespace FIPOPT::Sparse
{
    template <typename Derived, typename Vec, typename Observer>
    struct message_filter: public filter<message_filter<Derived, Vec>, Vec>
    {
        message_filter(std::vector<Observer*>& obs): observer_list_(obs){}

        inline filter_status Eval_Update_Filter(const double &alpha)
        {
            filter_status status = static_cast<Derived *>(this)->Eval_Update_Filter(alpha);
            Message(status, alpha);
            return status;
        }

        inline void Update_Direction(const Vec &x, const Vec &p)
        {
            return static_cast<Derived *>(this)->Update_Direction(x, p);
        }

        inline double Eval_theta(const double &alpha)
        {
            return static_cast<Derived *>(this)->Eval_theta(alpha);
        }

        inline double Eval_phi(const double &alpha)
        {
            return static_cast<Derived *>(this)->Eval_phi(alpha);
        }

        protected:
        inline void Message(const filter_status& s, const double& alpha)
        {
            for (auto observer: observer_list)
            {
                observer.Update(ID_, s, alpha);
            }
        }
    };
}

#endif