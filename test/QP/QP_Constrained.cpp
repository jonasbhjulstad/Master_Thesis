#include <Common/EigenDataTypes.hpp>
#include <Dense/Algorithms/LSFB/LSFB.hpp>
#include <Dense/Functors/Objective/Objective_QP/Objective_QP.hpp>
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

//Executable for linear-constrained QP
//fPath needs to be configured, along with .csv-parameters
int main()
{

    using Mat_x = Eigen::Matrix<double, Nx, Nx>;
    using Vec_x = Eigen::Matrix<double, Nx, 1>;
    using Mat_h = Eigen::Matrix<double, Nh, Nx>;
    using Vec_h = Eigen::Matrix<double, Nh, 1>;
    using Mat_g = Eigen::Matrix<double, Ng, Nx>;
    using Vec_g = Eigen::Matrix<double, Ng, 1>;
    using Vec_cI = Eigen::Matrix<double, Ng + 2 * Nx, 1>;
    using Vec_A = Eigen::Matrix<double, Nx + Nh, 1>;
    using Mat_A = Eigen::Matrix<double, Nx + Nh, Nx + Nh>;

    const std::string fPath = "/home/build/FIPOPT/Data/QP/";
    const std::string pPath = fPath + "Param/";
    Mat_x Q = load_csv<Mat_x>(pPath + "Q.csv");
    Vec_x c = load_csv<Vec_x>(pPath + "c.csv");
    Mat_h A = load_csv<Mat_h>(pPath + "A.csv");
    Vec_h b = load_csv<Vec_h>(pPath + "b.csv");
    Mat_g D = load_csv<Mat_g>(pPath + "D.csv");
    Vec_g e = load_csv<Vec_g>(pPath + "e.csv");
    Vec_x x_lb = load_csv<Vec_x>(pPath + "x_lb.csv");
    Vec_x x_ub = load_csv<Vec_x>(pPath + "x_ub.csv");
    Vec_x x0 = load_csv<Vec_x>(pPath + "x0.csv");
    const std::string output_path = fPath;
    const std::string journalist_ID = "journalist_f";

    // objective_QP<Nx, Ng, Nh> f(Q, c, A, b, D, e, x_lb, x_ub);

    objective_QP_memoized<Nx, Ng, Nh> f(Q, c, A, b, D, e, x_lb, x_ub);

    // observer_journalist<Nx, Ng, Nh> journalist_f(journalist_ID, output_path);
    // objective_QP_journalist<Nx, Ng, Nh> f(Q, c, A, b, D, e, x_lb, x_ub);

    LSFB<Eigen::HouseholderQR>::Solve(f, x0, fPath + "Trajectory/");
}