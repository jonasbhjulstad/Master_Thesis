#include <Sparse/Functors/Objective/Objective_SIF/Objective_SIF.hpp>
#include <Sparse/Algorithms/LSFB/LSFB.hpp>
#include <string>
const std::string fPath = "/home/deb/Documents/FIPOPT/Data/SIF/Problem/OUTSDIF.d";

using namespace FIPOPT::Sparse;
int main()
{


    objective_SIF f(fPath);

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

    dVec t = lbd0.topRows(Nh_);
    spMat hess = f.Eval_hessian(x0, t);

    std::cout << "Hessian:" << std::endl << hess << std::endl;


    // constrained_objective_QP f(Q, c, A, b, D, e);

    // Vec<2> x;
    // x << 2,3;
    // Vec<2> s;
    // s << 0,0;
    // std::cout << f.Eval_jac_g(x, s) << std::endl;

    // constrained_objective_QP g(Q, c, A, b);
    
    // std::cout << g.Eval_jac_g(x, s);

}