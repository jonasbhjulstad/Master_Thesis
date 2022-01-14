
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import sys
matplotlib.use
sys.path.append("/home/build/gitFIPOPT/Release/Plot/Nonlinear_QP/")
rootFolder = "/home/build/gitFIPOPT/"
pFolder_SOC = rootFolder + "Data/NC_QP/SOC/Trajectory/"
pFolder_NONSOC = rootFolder + "Data/NC_QP/NONSOC/Trajectory/"
figFolder = "/home/build/MT/figures/"

from Binder_NC_QP import *

def read_x_traj(objective_path, restoration_iters = []):
    mus = np.genfromtxt(objective_path + "mu.csv")
    xk_traj = []
    for mu in mus:
        xk_traj.append(np.genfromtxt(
                objective_path + "mu_{m:.6f}/x.csv".format(m=mu), delimiter=", ").reshape((-1,2)))
    x_sliced = [xk_traj[0]]
    for i, xk in enumerate(xk_traj[1:]):
        if i in restoration_iters:
            x_sliced.append([])
        x_sliced[-1] = np.concatenate([x_sliced[-1], xk], axis=0)
    return x_sliced


if __name__ == '__main__':


    mus = np.genfromtxt(pFolder_SOC + "mu.csv")
    mu = 100
    f = load_NC_QP()
    phi = load_barrier_NC_QP(f, mu)

    x_traj_SOC = np.concatenate(read_x_traj(pFolder_SOC), axis=0)
    x_traj_NONSOC = np.concatenate(read_x_traj(pFolder_NONSOC), axis=0)
    z_traj = np.genfromtxt(pFolder_SOC + "z.csv", delimiter=", ")

    min_x = np.min(x_traj_SOC, axis=0)
    max_x = np.max(x_traj_SOC, axis=0)
    plot_slack_x = (max_x-min_x)/10
    # x0 = np.linspace(min_x[0] - plot_slack_x[0], max_x[0] + plot_slack_x[0], 100)
    # x1 = np.linspace(min_x[1] - plot_slack_x[1], max_x[1] + plot_slack_x[1], 100)
    dx_prev = np.inf

    # Plot_name = "NC_QP"

    xs = x_traj_SOC[0,:]
    xf = x_traj_SOC[-1,:]
    dx = xf-xs
    xplot_s = xs - dx/4
    xplot_f = xf + dx/4
    if dx[0] != 0:
        x0 = np.linspace(xplot_s[0],xplot_f[0], 300)
    if dx[1] != 0:
        x1 = np.linspace(xplot_s[1], xplot_f[1], 300)

    # x0 = np.linspace(-10,10,100)
    # x1 = np.linspace(-10,10,100)
    X, Y = np.meshgrid(x0, x1)

    C = np.zeros(X.shape)
    Phi = np.zeros(X.shape)
    C_I = np.zeros((X.shape[0], X.shape[1], 2))
    grad_Phi = np.zeros(X.shape)
    x_ = list(zip(X.flatten(), Y.flatten()))
    C = np.reshape(list(map(lambda x: f(x), x_)), (X.shape))
    Phi = np.reshape(list(map(lambda x: phi(x), x_)), (X.shape))
    c_E = np.reshape(list(map(lambda x: f.Eval_h(x), x_)), (X.shape))

    a = f([2.,2.])

    x0_m = np.linspace(x0[0], x0[-1], 20)
    x1_m = np.linspace(x0[0], x1[-1], 20)
    X_markers, Y_markers = np.meshgrid(x0_m, x1_m)
    # Phis = [np.reshape(list(map(lambda x: p(x), x_)), (X.shape)) for p in phis]
    # grad_Phi = np.reshape(
    #     list(map(lambda x: np.linalg.norm(phi.Eval_grad(x), np.inf), x_)), (X.shape))

# C_I = np.reshape(list(map(lambda x: f.Eval_g(x), x_)),
#                  (X.shape[0], X.shape[1], 2))
# C[i, j] = f(x_[i,j,:])
# C_I[i,j,:] = f.Eval_g(x_[i,j,:])
# Phi[i,j] = C[i,j] - mu*np.sum(np.log(-C_I[i,j,:]))
# grad_Phi[i,j] = np.linalg.norm(phi.Eval_grad(x_[i,j,:]))

# Phi[np.isnan(Phi)] = 10e8
# Phi = Phi + np.abs(Phi.min().min())
# Phi = np.log(Phi)
# grad_Phi[np.isnan(grad_Phi)] = 1e10

    # iter = list(range(0, z_traj.shape[0]))
    # fig0, ax0 = plt.subplots()
    # ax0.plot(iter, x_traj, color='k')
    # ax0.plot(iter, z_traj, color='r')
    # ax0.set_yscale('log')

    # plt.show()
    # plt.plot([f(np.array([[1], [x]])) for x  =in x0])
    # plt.show()
    fig, ax = plt.subplots()

    # ax.axhline(y=x_lb[1])
    # ax.axvline(x=x_lb[0])

    from matplotlib import ticker
    # # ax.contour(X, Y, Phi)
    # [ax.contourf(X, Y, P, levels=[0, 1]) for P in Phis]
    ax.contourf(X, Y, Phi, cmap='Greys')
    # , cmap='Greys')#, levels=np.linspace(C.min().min(), C.max().max(), 10))
    CS = ax.contour(X, Y, C, cmap='Greys')
    ax.contour(X, Y, c_E, cmap='Greys', levels=[-1e-3, 1e-3])



    # ax.scatter(X_markers, Y_markers, marker='x', s=2, color='k')
    # ax.contourf(X, Y, , cmap='Greys')
    # , cmap='Greys')#, levels=np.linspace(C.min().min(), C.max().max(), 10))
    # CS = ax.contour(X, Y, C)
    ax.plot(x_traj_SOC[:,0], x_traj_SOC[:,1], color='k', marker='o')
    # ax.plot(x_traj_NONSOC[:,0], x_traj_NONSOC[:,1], color='k', marker='x', linestyle='dotted', label='SOC disabled')
    # ax.plot([1000, 10001], [1000, 1001], label=r'$c_E(x)$')
    ax.scatter(x_traj_SOC[0, 0], x_traj_SOC[0, 1], color='k', marker='x', s=.5)
    ax.scatter(x_traj_SOC[-1, 0], x_traj_SOC[-1, 1], color='k', marker='.')
    N_iter_SOC = x_traj_SOC.shape[0]
    N_iter_NONSOC = x_traj_NONSOC.shape[0]

    # ax.legend()
    ax.set_xlim(min(x0), max(x0))
    ax.set_ylim(min(x1), max(x1))
    name = 'Bounded'
    # plt.show()
    fig.savefig(figFolder + 'NC_QP_Trajectory.pdf')
        # fig.savefig()
