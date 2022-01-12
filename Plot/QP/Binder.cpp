#include <Common/EigenDataTypes.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Dense/Functors/Barrier/Logbarrier.hpp>
#include <Dense/Functors/Objective/Objective_QP/Objective_QP.hpp>
#include <Dense/Functors/Objective/Objective.hpp>
namespace py = pybind11;
using namespace FIPOPT::Dense;
template <int Row, int Col>
using Mat = Eigen::Matrix<double, Row, Col>;
using dMat = Eigen::MatrixXd;

template <int Nx, int Ng, int Nh>
logbarrier<objective_QP_base<Nx, Ng, Nh, objective>, Nx, Ng, Nh> load_barrier_QP(objective_QP<Nx, Ng, Nh> &f, const double &mu)
{
    using obj = objective_QP<Nx, Ng, Nh>;
    using QP_base = objective<obj, Nx, Ng, Nh>;
    logbarrier phi(f, mu);
    return phi;
}

template <int Nx, int Ng, int Nh>
const objective_QP<Nx, Ng, Nh> load_QP(const Mat<Nx, Nx> &Q, const Mat<Nx, 1> &c, const Mat<Nh, Nx> &A, const Mat<Nh, 1> &b, const Mat<Ng, Nx> &D, const Mat<Ng, 1> &e, const Mat<Nx, 1> &x_lb, const Mat<Nx, 1> &x_ub)
{
    objective_QP<Nx, Ng, Nh> f(Q, c, A, b, D, e, x_lb, x_ub);
    return f;
}

template <int Nx, int Ng, int Nh>
void declare_QP(auto &m)
{
    using base_QP = objective_QP_base<Nx, Ng, Nh>;
    using objective_barrier_QP = logbarrier<base_QP, Nx, Ng, Nh>;
    using Vec_x = Eigen::Matrix<double, Nx, 1>;
    using Vec_h = Eigen::Matrix<double, Nh, 1>;
    std::string name = "_" + std::to_string(Nx) + "_" + std::to_string(Ng) + "_" + std::to_string(Nh);

    py::class_<objective_barrier_QP>(m, ("objective_barrier_QP" + name).c_str())
        .def("__call__", &objective_barrier_QP::template operator()<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_grad", &objective_barrier_QP::template Eval_grad<Vec_x>, py::return_value_policy::reference_internal);

    m.def(("load_barrier_QP" + name).c_str(), &load_barrier_QP<Nx, Ng, Nh>);

    using objective_QP = objective_QP<Nx, Ng, Nh>;
    py::class_<objective_QP>(m, ("objective_QP" + name).c_str())
        .def("__call__", &objective_QP::template operator()<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_grad", &objective_QP::template Eval_grad<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_hessian_f", &objective_QP::template Eval_hessian_f<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_h", &objective_QP::template Eval_h<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_grad_h", &objective_QP::template Eval_grad_h<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_hessian_h", &objective_QP::template Eval_hessian_h<Vec_x, Vec_h>, py::return_value_policy::reference_internal)
        .def("Eval_g", &objective_QP::template Eval_g<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_grad_g", &objective_QP::template Eval_g<Vec_x>, py::return_value_policy::reference_internal)
        .def("Get_x_ub", &objective_QP::Get_x_ub, py::return_value_policy::reference_internal)
        .def("Get_x_lb", &objective_QP::Get_x_lb, py::return_value_policy::reference_internal);
    m.def(("load_QP" + name).c_str(), &load_QP<Nx, Ng, Nh>);


}

template <int Nx, int Ng, int Nh>
void declare_Inequality_Restoration_QP(auto &m)
{
    using base_QP = objective_QP_base<Nx, Ng, Nh>;
    using objective_barrier_QP = logbarrier<base_QP, Nx, Ng, Nh>;
    using Vec_x = Eigen::Matrix<double, Nx, 1>;
    using Vec_h = Eigen::Matrix<double, Nh, 1>;
    std::string name = "_" + std::to_string(Nx) + "_" + std::to_string(Ng) + "_" + std::to_string(Nh);

    py::class_<objective_barrier_QP>(m, ("objective_barrier_QP" + name).c_str())
        .def("__call__", &objective_barrier_QP::template operator()<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_grad", &objective_barrier_QP::template Eval_grad<Vec_x>, py::return_value_policy::reference_internal);

    m.def(("load_barrier_QP" + name).c_str(), &load_barrier_QP<Nx, Ng, Nh>);

    using objective_QP = objective_QP<Nx, Ng, Nh>;
    py::class_<objective_QP>(m, ("objective_QP" + name).c_str())
        .def("__call__", &objective_QP::template operator()<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_grad", &objective_QP::template Eval_grad<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_hessian_f", &objective_QP::template Eval_hessian_f<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_h", &objective_QP::template Eval_h<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_grad_h", &objective_QP::template Eval_grad_h<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_hessian_h", &objective_QP::template Eval_hessian_h<Vec_x, Vec_h>, py::return_value_policy::reference_internal)
        .def("Eval_g", &objective_QP::template Eval_g<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_grad_g", &objective_QP::template Eval_g<Vec_x>, py::return_value_policy::reference_internal)
        .def("Get_x_ub", &objective_QP::Get_x_ub, py::return_value_policy::reference_internal)
        .def("Get_x_lb", &objective_QP::Get_x_lb, py::return_value_policy::reference_internal);
    m.def(("load_QP" + name).c_str(), &load_QP<Nx, Ng, Nh>);
}

PYBIND11_MODULE(Binder_QP, m)
{
    //     py::class_<objective_barrier_QP>(m, "objective_barrier_QP")
    //         .def("__call__", &objective_barrier_QP::template operator()<dMat>, py::return_value_policy::reference_internal)
    //         .def("Eval_grad", &objective_barrier_QP::template Eval_grad<dMat>, py::return_value_policy::reference_internal);

    //     m.def("load_barrier_QP", &load_barrier_QP);
    // declare_QP<2,2,0>(m);
    declare_QP<2, 0, 0>(m);
    declare_QP<2, 1, 0>(m);
    declare_QP<2, 2, 0>(m);
    declare_QP<2, 3, 0>(m);
    declare_QP<2, 0, 1>(m);
    declare_QP<2, 0, 2>(m);
    declare_QP<2, 0, 3>(m);
    declare_QP<2, 1, 1>(m);
    declare_QP<2, 1, 2>(m);
    declare_QP<2, 1, 3>(m);
    declare_QP<2, 2, 1>(m);
    declare_QP<2, 2, 2>(m);
    declare_QP<2, 2, 3>(m);
    declare_QP<2, 3, 1>(m);
    declare_QP<2, 3, 2>(m);
    declare_QP<2, 3, 3>(m);
}