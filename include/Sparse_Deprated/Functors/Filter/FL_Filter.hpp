#ifndef FIPOPT_FL_FILTER_HPP
#define FIPOPT_FL_FILTER_HPP
// #include <cmath>
#include <iostream>
#include <Sparse/Functors/Barrier/Barrier.hpp>
#include <Sparse/Functors/Filter/Filter_Status.hpp>
#include <Common/Utils/Eigen_Utils.hpp>
namespace FIPOPT::Sparse
{

    struct FL_condition_pair
    {
        double theta = std::numeric_limits<double>::infinity();
        double phi = std::numeric_limits<double>::infinity();
    };
    struct FL_param
    {
        FL_param(const double &theta_min_, const double &theta_max_) : gamma_theta(1e-5), gamma_phi(1e-5), delta(1.), gamma_alpha(0.05),
                                                                       s_theta(1.1), s_phi(2.3), eta(1e-4), theta_max(theta_max_), theta_min(theta_min_), filter_set_size(100) {}

        double gamma_theta, gamma_phi, gamma_alpha;
        double delta;
        double eta;
        double theta_min, theta_max;
        double s_theta, s_phi;
        int filter_set_size;
    };
    template <typename Derived>
    struct FL_filter
    {
        barrier<Derived> &phi_;
        const FL_param P_;
        dVec x_;
        dVec p_;
        bool small_search_direction_;
        std::vector<FL_condition_pair> filter_set_{FL_condition_pair()};
        FL_filter(barrier<Derived> &phi,
                  const MatrixBase<spVec> &x,
                  const MatrixBase<spVec> &p) : x_(x),
                                                p_(p),
                                                phi_(phi),
                                                P_(FL_param(1e-4 * std::max(1., Eval_theta(0.)), 1e4 * std::max(1., Eval_theta(0.))))
        {
            filter_set_.reserve(P_.filter_set_size);
            Eval_Update_Condition_Pairs(0., filter_set_);
            small_search_direction_ = Eval_Small_Search_Direction(x, p);

        }

        inline filter_status Eval_Update_Filter(const double &alpha)
        {
            if (small_search_direction_)
                return FILTER_ACCEPTED;

            double t = Eval_theta(0.);
            double g = Eval_grad_phi(alpha);
            double p = Eval_phi(0.);
            bool small_theta = (Eval_theta(0.) <= P_.theta_min);
            bool negative_grad = Eval_grad_phi(alpha) < 0;

            double alpha_min = P_.gamma_alpha * P_.gamma_theta;
            // Feasibility Check

            if (negative_grad)
            {
                double descent_feasibility = Eval_Relative_Descent_Feasibility(alpha);
                alpha_min = (descent_feasibility == 0) ? alpha_min : std::min(alpha_min, descent_feasibility);
                if (small_theta)
                {
                    alpha_min = std::min(alpha_min, Eval_Switching_Feasibility(alpha));
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
            return l2_norm(phi_.Eval_cE(dVec(x_ + alpha * p_)));
        }

        inline double Eval_phi(const double &alpha)
        {
            return phi_(dVec(x_ + alpha * p_)).value();
        }

        inline double Eval_grad_phi(const double &alpha)
        {
            return Val(alpha * (phi_.Eval_grad(x_).transpose() * p_)).value();
        }

        inline bool Eval_Switching_Condition(const double &alpha)
        {
            bool grad_cond = Eval_grad_phi(1.0) < 0;
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

        inline void Update_Direction(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVec> &p)
        {
            x_ = x;
            p_ = p;
            small_search_direction_ = Eval_Small_Search_Direction(x, p);
        }

        inline bool Eval_Small_Search_Direction(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVec> &p)
        {
            return linf_norm(p.cwiseAbs() * (dVec(1. + dVec(x).array()).cwiseInverse())) < (10 * 1e-6);
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
            return pow(-Eval_grad_phi(1.), P_.s_phi);
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

        bool Eval_Update_Condition_Pairs(const double &alpha, std::vector<FL_condition_pair> &set)
        {
            const double theta = Eval_theta(alpha);
            const double phi = Eval_phi(alpha);
            if (std::isnan(theta) || std::isnan(phi))
                return false;

            for (auto &cond_pair : set)
            {
                bool theta_sufficient = Sufficient_Theta_Decrease(theta, cond_pair.theta);
                bool phi_sufficient = Sufficient_Phi_Decrease(phi, cond_pair);
                if (theta_sufficient && phi_sufficient)
                {
                    cond_pair = {theta, phi};
                    return true;
                }
                else if (!(theta_sufficient || phi_sufficient))
                {
                    return false;
                }
            }
            set.push_back({theta, phi});
            return true;
        }
    };
}

#endif