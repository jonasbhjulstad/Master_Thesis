
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import ticker as mticker
import sys
import os
matplotlib.use
sys.path.append("/home/deb/Documents/FIPOPT/Release/Plot/QP/")

QP_Folder = "/home/deb/Documents/FIPOPT/Data/QP/"
dFolder = QP_Folder + "Trajectory/"
pFolder = QP_Folder + "Param/"

from Binder_QP import *


def fix_dim_vec(mat):
    if len(mat) == 0:
        return np.ndarray(shape=(0,1), dtype=np.float64)
    else:
        return mat

def fix_dim_mat(mat):
    if len(mat) == 0:
        return np.ndarray(shape=(0,2), dtype=np.float64)
    else:
        return mat
def load_QP_params():

    Q = np.genfromtxt(pFolder + "Q.csv", delimiter=",")
    c = np.genfromtxt(pFolder + "c.csv", delimiter=",").reshape((-1,1))
    Nx = c.shape[0]
    A = np.genfromtxt(pFolder + "A.csv", delimiter=",")
    b = np.genfromtxt(pFolder + "b.csv", delimiter=",").reshape((-1,1))
    D = np.genfromtxt(pFolder + "D.csv", delimiter=",").reshape((1,-1))
    e = np.genfromtxt(pFolder + "e.csv", delimiter=",").reshape((-1,1))
    x_lb = np.genfromtxt(pFolder + "x_lb.csv", delimiter=",")
    x_ub = np.genfromtxt(pFolder + "x_ub.csv", delimiter=",")
    x0 = np.genfromtxt(pFolder + "x0.csv", delimiter=",").reshape((-1,1))
    # x_traj = np.genfromtxt(pFolder + "x_iter.csv", delimiter=",")
    # z_traj = np.genfromtxt(pFolder + "z_iter.csv", delimiter=",")
    return Q, fix_dim_vec(c), fix_dim_mat(A), fix_dim_vec(b), fix_dim_mat(D), fix_dim_vec(e), x_lb, x_ub, x0



rootFolder = "/home/deb/Documents/FIPOPT/"
sys.path.append(rootFolder + "build/test/Plot/")

figFolder = "/home/deb/Documents/FIPOPT/figures/"


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def read_x_traj(objective_path, restoration_iters = []):
    mus = np.genfromtxt(objective_path + "mu.csv").reshape((-1))
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

    mu = 1
    mu_list = np.genfromtxt(dFolder + "mu.csv")

    objvals = np.genfromtxt(dFolder + "obj.csv", delimiter=", ");
    Q, c, A, b, D, e, x_lb, x_ub, x0 = load_QP_params()
    Nx = str(Q.shape[0])
    Ng = str(D.shape[0])
    Nh = str(A.shape[0])
    mus = np.genfromtxt(dFolder + "mu.csv")
    f = eval("load_QP_{}_{}_{}(Q, c, A, b, D, e, x_lb, x_ub)".format(Nx, Ng, Nh))
    phi = eval("load_barrier_QP_{}_{}_{}(f, mu)".format(
        Nx, Ng, Nh))

    x_traj = np.genfromtxt(
        dFolder + "x.csv", delimiter=", ")
    z_traj = np.genfromtxt(
        dFolder + "z.csv", delimiter=", ")

    min_x = np.min(x_traj, axis=0)
    max_x = np.max(x_traj, axis=0)

    plot_slack_x = (max_x-min_x)/10
    # x0 = np.linspace(min_x[0] - plot_slack_x[0], max_x[0] + plot_slack_x[0], 100)
    # x1 = np.linspace(min_x[1] - plot_slack_x[1], max_x[1] + plot_slack_x[1], 100)
    dx_prev = np.inf

    fig, ax = plt.subplots()

    x_cp_traj = np.genfromtxt(dFolder + "x.csv", delimiter=",")
    xk_traj = x_cp_traj[0,:].reshape((1,2))
    xmax = np.full(xk_traj.shape, -np.inf)
    xmin = -xmax
    restoration_iters = []
    ineq_dirs, eq_dirs, ineq_x_trajs, eq_x_trajs = [], [], [], []
    for filename in os.listdir(dFolder):
        if filename.startswith("Inequality_Restoration"): 
            ineq_dirs.append(filename + "/")
            restoration_iters.append(filename[-1]) 
        elif filename.startswith("Equality_Restoration"): 
            eq_dirs.append(filename + "/")
            restoration_iters.append(filename[-1])


    for idir in ineq_dirs:
        cI_x = np.genfromtxt(dFolder + idir + "x.csv", delimiter=",")
        cI_x = np.reshape(cI_x, (-1, cI_x.shape[-1]))[:,:2]
        xmin = np.min(np.concatenate([xmin, cI_x], axis=0), axis=0).reshape((1,-1))
        xmax = np.max(np.concatenate([xmax, cI_x], axis=0), axis=0).reshape((1,-1))
        # ax.scatter(cI_x[0,0], cI_x[0,1], marker='.', color='k')
        # ax.scatter(cI_x[-1,0], cI_x[-1,1], marker='.', color='k')
        # ax.plot([cI_x[0,0], cI_x[-1,0]], [cI_x[0,1], cI_x[-1,1]], linestyle='dotted', color='k', label='Inequality Restoration');
        
        ax.plot(cI_x[:,0], cI_x[:,1], linestyle='dotted', color='k', label='Inequality Restoration');
    # for edir in eq_dirs:
    #     cE_x = np.genfromtxt(dFolder + edir + "x.csv", delimiter=",")[:,:2]
    #     xmin = np.min(np.concatenate([xmin, cE_x], axis=0), axis=0).reshape((1,-1))
    #     xmax = np.max(np.concatenate([xmax, cE_x], axis=0), axis=0).reshape((1,-1))
    #     ax.scatter(cE_x[0,0], cE_x[0,1], marker='.', color='k')
    #     ax.scatter(cE_x[-1,0], cE_x[-1,1], marker='.', color='k')
    #     # ax.plot([cE_x[0,0], cE_x[-1,0]], [cE_x[0,1], cE_x[-1,1]], linestyle='dotted', color='k', label='Inequality Restoration');
    #     ax.plot(cE_x[:,0], cE_x[:,1], linestyle='dashed', color='k', label='Equality Restoration');
    main_x_traj = read_x_traj(dFolder)
    # ineq_traj = read_x_traj(dFolder + "Inequality_Restoration_0/")
    xmin = np.min(np.concatenate([xmin, np.concatenate(main_x_traj, axis=0)], axis=0), axis=0)
    xmax = np.max(np.concatenate([xmax, np.concatenate(main_x_traj, axis=0)], axis=0), axis=0)



    phi = eval(
        "load_barrier_QP_{}_{}_{}(f, mus[-1])".format(Nx, Ng, Nh))
    xplot_s = xmin - (xmax-xmin)/2
    xplot_f = xmax + (xmax-xmin)/2



    x0 = np.linspace(xplot_s[0], xplot_f[0], 300)
    x1 = np.linspace(xplot_s[1], xplot_f[1], 300)
    x0_m = np.linspace(xplot_s[0], xplot_f[0], 20)
    x1_m = np.linspace(xplot_s[1], xplot_f[1], 20)
    X_markers, Y_markers = np.meshgrid(x0_m, x1_m)

    X, Y = np.meshgrid(x0, x1)

    C = np.zeros(X.shape)
    Phi = np.zeros(X.shape)
    C_E = np.zeros((X.shape[0], X.shape[1], 2))
    grad_Phi = np.zeros(X.shape)
    x_ = list(zip(X.flatten(), Y.flatten()))
    C = np.reshape(list(map(lambda x: f(x), x_)), (X.shape))
    C_E = np.reshape(list(map(lambda x: np.linalg.norm(f.Eval_h(x)), x_)), (X.shape))
    Phi = np.reshape(list(map(lambda x: phi(x), x_)), (X.shape))

    ax.scatter(X_markers, Y_markers, marker='x', s=2, color='k')
    ax.contourf(X, Y, Phi, cmap='Greys')
    ax.contour(X, Y, C_E, levels=[0])
    # , cmap='Greys')#, levels=np.linspace(C.min().min(), C.max().max(), 10))
    CS = ax.contour(X, Y, C, cmap='Greys')
    ax.set_xlim(xplot_s[0], xplot_f[0])
    ax.set_ylim(xplot_s[1], xplot_f[1])
    # ax.contourf(X, Y, grad_Phi, levels = np.linspace(1e-6,1000,10), colors='k')
    # ax.contour(X, Y, C_I[:, :, 0], levels=[0], colors='k')
    # ax.contour(X, Y, C_I[:, :, 1], levels=[0], colors='k')
    # ax.contourf(X, Y, grad_Phi)
    ax.scatter(xk_traj[0, 0], xk_traj[0, 1], color='k', marker='x', s=.5)
    ax.scatter(xk_traj[-1, 0], xk_traj[-1, 1], color='k', marker='.')


    from matplotlib.ticker import ScalarFormatter

    y_formatter = ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)

    # ax.scatter(x_cp_traj[0,0], x_cp_traj[0,0], color='k', marker='.')
    # # ax.plot(x_ineq_cp_traj[:,0], x_ineq_cp_traj[:,1], color='k', linestyle='dotted', label='Restoration')
    # ax.scatter(x_cp_traj[1,0], x_cp_traj[1,0], color='k', marker='.')
    for xm in main_x_traj:
        ax.plot(xm[:,0], xm[:,1], color='k', label='Barrier Subproblem')
    ax.scatter(x_cp_traj[-1,0], x_cp_traj[-1,1], color='k', marker='^', label=r"$x^*$")


    ax.set_title("$x^* = [{:.1f}, {:.1f}]$, $f^* = {:.3e}$, $E_0 = {:.3e}$, $N_\mu = {}$".format(x_cp_traj[-1][0], x_cp_traj[-1][1], objvals[-1,1], objvals[-1,0], len(mu_list)))
    # plt.show()
    fig.subplots_adjust(hspace=1.)
    fig.subplots_adjust(wspace=1.)
    plt.show()
    # fig.savefig(figFolder + "QP_feasible" + "_Trajectory", format='eps')
    # plt.show()
