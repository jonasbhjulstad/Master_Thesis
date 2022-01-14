#include <Common/EigenDataTypes.hpp>
#include <Dense/Algorithms/LSFB/LSFB.hpp>
#include <Dense/Functors/Objective/Objective_QP/NC_QP.hpp>
// #include <Dense/Functors/Objective/Objective_QP/NE_QP.hpp>
#include <iostream>
#include <Dense/Algorithms/LSFB/LSFB.hpp>
#include <Dense/Initial_Multipliers/Initial_Multipliers.hpp>
#include <Common/Utils/EigenParser.hpp>
#include <SIF_Dimensions/Dimensions.hpp>
constexpr static int Nx = 2;
constexpr static int Ng = 1;
constexpr static int Nh = 0;
using namespace FIPOPT::Dense;
template <typename T>
using LinSolver = Eigen::HouseholderQR<T>;

//Nonlinear-constrained Quadratic Program
//Used to solve nonlinear inequality/equality QPs by including NC_QP.hpp OR NE_QP.hpp
int main()
{

    using Vec_x = Eigen::Matrix<double, Nx, 1>;

    const std::string fPath = "//home/build/FIPOPT/Data/NC_QP/SOC/";
    Vec_x x0 = load_csv<Vec_x>(fPath + "x0.csv");

    NC_QP f;

    LSFB<Eigen::HouseholderQR>::Solve(f, x0, fPath + "Trajectory/");
}