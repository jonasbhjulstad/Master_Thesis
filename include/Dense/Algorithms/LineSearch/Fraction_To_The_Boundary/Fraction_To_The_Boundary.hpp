#ifndef FIPOPT_FRACTION_TO_THE_BOUNDARY_HPP
#define FIPOPT_FRACTION_TO_THE_BOUNDARY_HPP
#include <Common/EigenDataTypes.hpp>
namespace FIPOPT::Dense
{
    template <typename Vec>
    inline double Solve_FTTB_Step(
        const MatrixBase<Vec> &z,
        const MatrixBase<Vec> &d_z,
        const double &tau_j)
    {
        Eigen::Matrix<double, Vec::RowsAtCompileTime, 1> alphas = (-tau_j * z.cwiseProduct(d_z.cwiseInverse()).array()).matrix();
        alphas = (alphas.array() >= 0).select(alphas, 1e8);
        double alpha_min = alphas.minCoeff();
        return std::max(std::min(alpha_min, 1.0), 0.);
    }

    inline void Update_FTTB_tau(const double &mu, double &tau)
    {
        tau = std::max(.99, 1 - mu);
    }

}
#endif