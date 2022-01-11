#ifndef FIPOPT_BS_JOURNALIST_DENSE_HPP
#define FIPOPT_BS_JOURNALIST_DENSE_HPP
#include <fstream>
#include <Common/Utils/Print.hpp>
#include <Dense/Optimality/Optimality.hpp>
#include <Dense/Functors/Objective/Objective.hpp>

namespace FIPOPT::Dense
{
    struct BS_iteration_journalist
    {
        BS_iteration_journalist(const std::string &dir, const double &s_max) : s_max_(s_max)
        {
            namespace fs = std::filesystem;
            fs::create_directories(dir);
            f_x.open(dir + "x.csv");
            f_lbd.open(dir + "lbd.csv");
            f_z.open(dir + "z.csv");
            f_obj.open(dir + "obj.csv");
            f_theta.open(dir + "theta.csv");
            f_alpha.open(dir + "alpha.csv");

        }

        template <typename Derived, int Nx, int Ng, int Nh, typename Vec_x, typename Vec_cE, typename Vec_cI>
        void Write(objective<Derived, Nx, Ng, Nh> &f, const Vec_x &x, const Vec_cE &lbd, const Vec_cI &z,
                          const double &alpha, const double &alpha_z_g, const double &alpha_z_ub, const double &alpha_z_lb)
        {
            const Eigen::IOFormat CSVFormat = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
            f_x << x.transpose().format(CSVFormat) << '\n';
            f_lbd << lbd.transpose().format(CSVFormat) << '\n';
            f_z << z.transpose().format(CSVFormat) << '\n';
            f_obj << Eval_Global_Optimality_Error(f, x, lbd, z, s_max_) << ", " << f(x) << '\n';
            f_theta << f.Eval_cE(x).norm() << '\n';
            f_alpha << alpha << ", " << alpha_z_g << ", " << alpha_z_g << ", " << alpha_z_ub << ", " << alpha_z_lb << '\n';
        }

        ~BS_iteration_journalist()
        {
            f_x.close();
            f_lbd.close();
            f_z.close();
            f_obj.close();
            f_theta.close();
            f_alpha.close();

        }

    private:
        std::ofstream f_x;
        std::ofstream f_z;
        std::ofstream f_lbd;

        std::ofstream f_obj;
        std::ofstream f_theta;
        std::ofstream f_alpha;

        const double s_max_;
    };
}

#endif