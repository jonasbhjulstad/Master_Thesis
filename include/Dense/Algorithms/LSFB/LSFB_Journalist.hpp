#ifndef FIPOPT_LSFB_Journalist_DENSE_HPP
#define FIPOPT_LSFB_Journalist_DENSE_HPP
#include <fstream>
#include <Common/Utils/Print.hpp>
#include <Dense/Optimality/Optimality.hpp>
#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Algorithms/Barrier_Subproblem/BS_Journalist.hpp>

namespace FIPOPT::Dense
{

    struct LSFB_iteration_journalist
    {
        LSFB_iteration_journalist(const std::string &pwd, const double &s_max) : s_max_(s_max)
        {
            namespace fs = std::filesystem;
            fs::create_directories(pwd);
            f_x.open(pwd + "x.csv");
            f_lbd.open(pwd + "lbd.csv");
            f_z.open(pwd + "z.csv");
            f_mu.open(pwd + "mu.csv");
            f_obj.open(pwd + "obj.csv");
            f_theta.open(pwd + "theta.csv");
        }

        template <typename Derived, int Nx, int Ng, int Nh, typename Vec_x, typename Vec_cE, typename Vec_cI>
        inline void Write(objective<Derived, Nx, Ng, Nh> &f, const Vec_x &x, const Vec_cE &lbd, const Vec_cI &z, const double &mu)
        {
            const Eigen::IOFormat CSVFormat = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
            f_x << x.transpose().format(CSVFormat) << '\n';
            f_lbd << lbd.transpose().format(CSVFormat) << '\n';
            f_z << z.transpose().format(CSVFormat) << '\n';
            f_mu << mu << '\n';
            f_obj << Eval_Global_Optimality_Error(f, x, lbd, z, s_max_) << ", " << f(x) << '\n';
            f_theta << f.Eval_cE(x).norm() << '\n';
        }



        ~LSFB_iteration_journalist()
        {
            f_x.close();
            f_lbd.close();
            f_z.close();
            f_mu.close();
            f_obj.close();
            f_theta.close();
            f_mu.close();
        }

    private:
        std::ofstream f_x;
        std::ofstream f_z;
        std::ofstream f_lbd;

        std::ofstream f_mu;
        std::ofstream f_obj;
        std::ofstream f_theta;
        std::ofstream f_next_BS;

        const double s_max_;
    };
}

#endif