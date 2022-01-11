#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Barrier/Logbarrier.hpp>
#include <Dense/Functors/Objective/Objective_QP/NE_QP.hpp>
#include <Dense/Functors/Objective/Objective.hpp>
namespace py = pybind11;
using namespace FIPOPT::Dense;
template <int Row, int Col>
using Mat = Eigen::Matrix<double, Row, Col>;
using dMat = Eigen::MatrixXd;

constexpr static int Nx = 2;
constexpr static int Ng = 0;
constexpr static int Nh = 1;

NC_QP load_NC_QP()
{
    NC_QP f;
    return f;
}

logbarrier<NC_QP, Nx, Ng, Nh> load_barrier_NC_QP(NC_QP &f, const double &mu)
{
    logbarrier phi(f, mu);
    return phi;
}

PYBIND11_MODULE(Binder_NC_QP, m)
{
    using barrier_NC_QP = logbarrier<NC_QP, Nx, Ng, Nh>;

    using Vec_x = Eigen::Matrix<double, Nx, 1>;
    using Vec_h = Eigen::Matrix<double, Nh, 1>;
    py::class_<barrier_NC_QP>(m, "barrier_NC_QP")
        .def("__call__", &barrier_NC_QP::template operator()<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_grad", &barrier_NC_QP::template Eval_grad<Vec_x>, py::return_value_policy::reference_internal);
        m.def("load_barrier_NC_QP", &load_barrier_NC_QP);
        m.def("load_NC_QP", &load_NC_QP);


    py::class_<NC_QP>(m, "NC_QP")
        .def("__call__", &NC_QP::template operator()<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_grad", &NC_QP::template Eval_grad<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_hessian_f", &NC_QP::template Eval_hessian_f<Vec_x>, py::return_value_policy::reference_internal)
        .def("Eval_h", &NC_QP::template Eval_h<Vec_x>, py::return_value_policy::reference_internal)
        // .def("Eval_grad_h", &NC_QP::template Eval_grad_h<Vec_x>, py::return_value_policy::reference_internal)
        // .def("Eval_hessian_h", &NC_QP::template Eval_hessian_h<Vec_x, Vec_h>, py::return_value_policy::reference_internal)
        // .def("Eval_g", &NC_QP::template Eval_g<Vec_x>, py::return_value_policy::reference_internal)
        // .def("Eval_grad_g", &NC_QP::template Eval_g<Vec_x>, py::return_value_policy::reference_internal)
        .def("Get_x_ub", &NC_QP::Get_x_ub, py::return_value_policy::reference_internal)
        .def("Get_x_lb", &NC_QP::Get_x_lb, py::return_value_policy::reference_internal);
}