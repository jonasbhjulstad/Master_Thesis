#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Common/Cutest.hpp>
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Barrier/Logbarrier.hpp>
#include <Dense/Functors/Objective/Objective_SIF/Objective_SIF.hpp>
#include <Dense/Functors/Objective/Objective.hpp>
#include <string>
namespace py = pybind11;
using namespace FIPOPT::Dense;
template <int Row, int Col>
using Mat = Eigen::Matrix<double, Row, Col>;
using dMat = Eigen::MatrixXd;

template <int Nx, int Ng, int Nh>
logbarrier<objective_SIF_base<Nx, Ng, Nh, objective>, Nx, Ng, Nh> load_logbarrier_SIF(objective_SIF<Nx, Ng, Nh> &f, const double &mu)
{
    logbarrier phi(f, mu);
    return phi;
}

template <int Nx, int Ng, int Nh>
const objective_SIF<Nx, Ng, Nh> load_SIF(const std::string& fPath)
{
    objective_SIF<Nx, Ng, Nh> f(fPath);
    return f;
}


template <int Nx, int Ng, int Nh>
void declare_SIF(auto& m)
    {
        using objective_SIF = objective_SIF<Nx, Ng, Nh>;
        using base_SIF = objective_SIF_base<Nx, Ng, Nh>;
        using objective_barrier_SIF = logbarrier<base_SIF, Nx, Ng, Nh>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Vec_h = Eigen::Matrix<double, Nh, 1>;
        std::string name = "_" + std::to_string(Nx) + "_" + std::to_string(Ng) + "_" + std::to_string(Nh);

        py::class_<objective_barrier_SIF>(m, ("objective_barrier_SIF" + name).c_str())
            .def("__call__", &objective_barrier_SIF::template operator()<Vec_x>, py::return_value_policy::reference_internal)
            .def("Eval_grad", &objective_barrier_SIF::template Eval_grad<Vec_x>, py::return_value_policy::reference_internal);

        m.def(("load_barrier_SIF" + name).c_str(), &load_logbarrier_SIF<Nx, Ng, Nh>);

        py::class_<objective_SIF>(m, ("objective_SIF" + name).c_str())
            .def("__call__", &objective_SIF::template operator()<Vec_x>, py::return_value_policy::reference_internal)
            .def("Eval_grad", &objective_SIF::template Eval_grad<Vec_x>, py::return_value_policy::reference_internal)
            .def("Eval_hessian_f", &objective_SIF::template Eval_hessian_f<Vec_x>, py::return_value_policy::reference_internal)
            .def("Eval_h", &objective_SIF::template Eval_h<Vec_x>, py::return_value_policy::reference_internal)
            .def("Eval_grad_h", &objective_SIF::template Eval_grad_h<Vec_x>, py::return_value_policy::reference_internal)
            .def("Eval_hessian_h", &objective_SIF::template Eval_hessian_h<Vec_x, Vec_h>, py::return_value_policy::reference_internal)
            .def("Eval_g", &objective_SIF::template Eval_g<Vec_x>, py::return_value_policy::reference_internal)
            .def("Eval_grad_g", &objective_SIF::template Eval_g<Vec_x>, py::return_value_policy::reference_internal)
            .def("Get_x_ub", &objective_SIF::Get_x_ub, py::return_value_policy::reference_internal)
            .def("Get_x_lb", &objective_SIF::Get_x_lb, py::return_value_policy::reference_internal);
        m.def(("load_SIF" + name).c_str(), &load_SIF<Nx, Ng, Nh>);
    }

PYBIND11_MODULE(Binder_SIF, m)
{
    //     py::class_<objective_barrier_SIF>(m, "objective_barrier_SIF")
    //         .def("__call__", &objective_barrier_SIF::template operator()<dMat>, py::return_value_policy::reference_internal)
    //         .def("Eval_grad", &objective_barrier_SIF::template Eval_grad<dMat>, py::return_value_policy::reference_internal);

    //     m.def("load_barrier_SIF", &load_barrier_SIF);
    // declare_SIF<2,2,0>(m);
    declare_SIF<2,0,0>(m);
    declare_SIF<2,1,0>(m);
    declare_SIF<2,2,0>(m);
    declare_SIF<2,3,0>(m);
    declare_SIF<2,0,1>(m);
    declare_SIF<2,0,2>(m);
    declare_SIF<2,0,3>(m);
    declare_SIF<2,1,1>(m);
    declare_SIF<2,1,2>(m);
    declare_SIF<2,1,3>(m);
    declare_SIF<2,2,1>(m);
    declare_SIF<2,2,2>(m);
    declare_SIF<2,2,3>(m);
    declare_SIF<2,3,1>(m);
    declare_SIF<2,3,2>(m);
    declare_SIF<2,3,3>(m);
}