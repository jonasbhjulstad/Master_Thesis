#include <Common/EigenDataTypes.hpp>
#include <Common/Cutest.hpp>
#include <Dense/Functors/Objective/Objective_SIF/Objective_SIF.hpp>
#include <Dense/Algorithms/LSFB/LSFB.hpp>
// #include <Dense/Functors/Objective/Observer_Journalist/Observer_Journalist.hpp>
#include <SIF_Dimensions/Dimensions.hpp>

//Executable for all SIF-problems
//fPath, dPath needs to be configured

constexpr static int Nx = GLOBAL_DIM_X;
constexpr static int Ng = GLOBAL_DIM_G;
constexpr static int Nh = GLOBAL_DIM_H;
using namespace FIPOPT::Dense;
template <typename T>
using LinSolver = Eigen::PartialPivLU<T>;
std::string load_SIF_name(const std::string&);


int main()
{

    using Objective = objective_SIF_journalist<Nx, Ng, Nh>;
    using Mat_A = Eigen::Matrix<double, Nx + Nh, Nx + Nh>;
    using Vec_x = Eigen::Matrix<double, Nx, 1>;

    const std::string SIF_path = "/home/deb/Documents/FIPOPT/Data/SIF/Problem/";
    const std::string SIFname = load_SIF_name(SIF_path + "probname.txt");
    const std::string dPath = "/home/deb/Documents/FIPOPT/Data/SIF/HS/";
    const std::string output_path = dPath + SIFname + "/";
    const std::string journalist_ID = "journalist_f";

    // objective_SIF<Nx, Ng, Nh> f(SIF_path + "OUTSDIF.d");

    objective_SIF_memoized<Nx, Ng, Nh> f(SIF_path + "OUTSDIF.d");

    // observer_journalist<Nx, Ng, Nh> journalist_f(journalist_ID, output_path);
    // objective_SIF_journalist<Nx, Ng, Nh> f(SIF_path + "OUTSDIF.d", journalist_f);
    
    std::ofstream file(dPath + SIFname + "/success.txt");
    Vec_x x0 = f.Get_x0();
    if (LSFB<LinSolver>::Solve(f, x0, dPath + SIFname + "/") == LSFB_ACCEPTED)
    {
        std::cout << SIFname << " solved successfully" << std::endl;
        file << "1";
    }
    else
    {
        file << "0"; 
    }
    file.close();
    return 0;
}

std::string load_SIF_name(const std::string& fPath)
{
    std::ifstream f_SIFname(fPath);
    std::string SIFname;
    f_SIFname >> SIFname;
    f_SIFname.close();
    return SIFname;
}