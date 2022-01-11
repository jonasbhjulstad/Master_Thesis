#ifndef FIPOPT_FL_FILTER_HPP
#define FIPOPT_FL_FILTER_HPP
// #include <cmath>
#include <iostream>
#include <Dense/Functors/Barrier/Barrier.hpp>
#include <Dense/Functors/Filter/Filter_Status.hpp>
#include <Dense/Functors/Filter/FL_Param.hpp>
namespace FIPOPT::Dense
{
    template <typename Derived, int Nx, int Ng, int Nh>
    struct FL_filter
    {
        using Objective_Barrier = barrier<Derived, Nx, Ng, Nh>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;

        Objective_Barrier &phi_;
        const FL_param P_;
        Vec_x x_;
        Vec_x p_;
        bool small_search_direction_;
        FL_filter_set& filter_set_;
        FL_filter(Objective_Barrier &phi,
                  const MatrixBase<Vec_x> &x,
                  FL_filter_set& filter_set) : x_(x),
                                                phi_(phi),
                                                P_(FL_param(1e-4 * std::max(1., Eval_theta(0.)), 1e4 * std::max(1., Eval_theta(0.)))),
                                                filter_set_(filter_set)
        {
            Eval_Update_Condition_Pairs(0., filter_set_);
        }

        FL_filter(FL_filter<Derived, Nx, Ng, Nh> &&F_old,
                  Objective_Barrier &phi,
                  const MatrixBase<Vec_x> &x,
                  const MatrixBase<Vec_x> &p):  x_(x),
                                                                p_(p),
                                                                phi_(phi),
                                                                P_(F_old.P_),
                                                                filter_set_(F_old.filter_set_)
                {
                    small_search_direction_ = Vec_x(p.array().abs() * ((1. + x.array()).inverse())).template lpNorm<Eigen::Infinity>() < (10 * 1e-4);
                }

            inline filter_status Eval_Update_Filter(const double &alpha)
        {
            if (small_search_direction_)
                return FILTER_ACCEPTED;

            bool small_theta = (Eval_theta(0.) <= P_.theta_min);
            bool negative_grad = Eval_grad_phi(alpha) < 0;

            double alpha_min = P_.gamma_alpha * P_.gamma_theta;
            // Feasibility Check

            if (std::isinf(Eval_phi(0.)))
            {
                return FILTER_REJECTED;
            }
            if constexpr (Nh > 0)
            {
                if (negative_grad)
                {
                    double descent_feasibility = Eval_Relative_Descent_Feasibility(alpha);
                    alpha_min = (descent_feasibility == 0) ? alpha_min : std::min(alpha_min, descent_feasibility);
                    if (small_theta && Nh != 0)
                    {
                        alpha_min = std::min(alpha_min, Eval_Switching_Feasibility(alpha));
                    }
                }
            }
            if (alpha < alpha_min)
            {
                return FILTER_INFEASIBLE_STEP_SIZE;
            }

            // Filter acceptance
            if (small_theta)
            {
                if (Eval_Switching_Condition(alpha) && Eval_Armijo_Condition(alpha))
                    return FILTER_ACCEPTED;
            }
            else if (Eval_Update_Condition_Pairs(alpha, filter_set_))
            {
                return FILTER_ACCEPTED;
            }

            return FILTER_REJECTED;
        }

        // Linesearch:
        inline double Eval_theta(const double &alpha)
        {
            return phi_.Eval_cE(Vec_x(x_ + alpha * p_)).norm();
        }

        inline double Eval_phi(const double &alpha)
        {
            return phi_(Vec_x(x_ + alpha * p_)).value();
        }

        inline double Eval_grad_phi(const double &alpha)
        {
            return Val(alpha * (phi_.Eval_grad(x_).transpose() * p_)).value();
        }

        inline bool Eval_Switching_Condition(const double &alpha)
        {
            bool grad_cond = Eval_grad_phi(1.0) < 0;
            double phip = Phi_Power_Condition();
            double thetap = Theta_Power_Condition();
            bool pow_cond = (alpha * Phi_Power_Condition()) > Theta_Power_Condition();
            return grad_cond && pow_cond;
        }
        inline bool Eval_Armijo_Condition(const double &alpha)
        {
            double p_alpha = Eval_phi(alpha);
            double p_grad = Eval_grad_phi(alpha);
            double rhs = (Eval_phi(0.) + P_.eta * Eval_grad_phi(alpha));

            return Eval_phi(alpha) <= (Eval_phi(0.) + P_.eta * Eval_grad_phi(alpha));
        }

        // Copy assignments:

        inline void Update_Direction(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_x> &p)
        {
            x_ = x;
            p_ = p;
            small_search_direction_ = Vec_x(p.array().abs() * ((1. + x.array()).inverse())).template lpNorm<Eigen::Infinity>() < (10 * 1e-6);
        }

    private:
        // Feasibility conditions

        inline double Eval_Switching_Feasibility(const double &alpha)
        {
            return Theta_Power_Condition() / Phi_Power_Condition();
        }

        inline double Eval_Relative_Descent_Feasibility(const double &alpha)
        {
            return -P_.gamma_phi * Eval_theta(0.) / Eval_grad_phi(1);
        }

        // Descent conditions

        inline double Theta_Power_Condition()
        {
            return P_.delta * pow(Eval_theta(0), P_.s_theta);
        }

        inline double Phi_Power_Condition()
        {
            return pow(-phi_.Eval_grad(x_).transpose() * p_, P_.s_phi);
        }

        // Filter-pair conditions

        bool Sufficient_Theta_Decrease(const double &theta, const double &theta_old)
        {
            return theta < (1 - P_.gamma_theta) * theta_old;
        }

        bool Sufficient_Phi_Decrease(const double &phi, const FL_condition_pair &pair)
        {
            return (phi < (pair.phi - P_.gamma_phi * pair.theta));
        }

        bool Eval_Update_Condition_Pairs(const double &alpha, FL_filter_set &set)
        {
            const double theta = Eval_theta(alpha);
            const double phi = Eval_phi(alpha);
            if (std::isnan(theta) || std::isnan(phi))
                return false;
            bool theta_sufficient = true;
            bool phi_sufficient = true;

            for (auto &cond_pair : set)
            {
                theta_sufficient = theta_sufficient && Sufficient_Theta_Decrease(theta, cond_pair.theta);
                phi_sufficient = phi_sufficient && Sufficient_Phi_Decrease(phi, cond_pair);
                if (theta_sufficient && phi_sufficient)
                {
                    cond_pair = {theta, phi};
                    return true;
                }
                if (!(theta_sufficient || phi_sufficient))
                    return false;
            }
            set.push_back({theta, phi});
            return true;
        }
    };
    template <typename Derived, int Nx, int Ng, int Nh>
    FL_filter(barrier<Derived, Nx, Ng, Nh> &phi,
              const MatrixBase<Eigen::Matrix<double, Nx, 1>> &x,
              const MatrixBase<Eigen::Matrix<double, Nx, 1>> &p) -> FL_filter<Derived, Nx, Ng, Nh>;
}

#endif