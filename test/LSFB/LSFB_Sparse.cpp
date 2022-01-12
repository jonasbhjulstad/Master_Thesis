#include <Common/Cutest.hpp>
#include <Sparse/Functors/Objective/Objective_SIF/Objective_SIF.hpp>
#include <Sparse/Algorithms/LSFB/LSFB.hpp>
#include <Eigen/SparseQR>
#include <string>
const std::string fPath = "/home/deb/Documents/gitFIPOPT/Data/SIF/Problem/";

using namespace FIPOPT::Sparse;
int main()
{
    objective_SIF f(fPath + "OUTSDIF.d");

    dVec x0 = f.Get_x0();
    std::cout << x0<< std::endl;

    dVec lbd0 = f.Get_lbd0();
    int Nh_ = f.Get_Nh();
    // x << 10.0, 0.5, 1000, 20;


    std::cout << "f(x):";
    std::cout << f(x0) << std::endl;


    dVec grad_f = f.Eval_grad(x0);

    std::cout << "Gradient:" << std::endl;
    std::cout << grad_f << std::endl;

    dVec t = lbd0.head(Nh_);
    spMat hess = f.Eval_hessian(x0, t);

    std::cout << "Hessian:" << std::endl << hess << std::endl;
    Eigen::SparseQR<spMat, Eigen::AMDOrdering<int>> solver;
    
    Solve_LSFB(f, x0, solver, fPath);
}