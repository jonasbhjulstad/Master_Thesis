
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker as mticker

import sys
rootFolder = "/home/build/FIPOPT/"
sys.path.append(rootFolder + "build/Plot/SIF/")

from Binder_SIF import *
pFolder = rootFolder + "Data/SIF/Problem/"
outsdif = pFolder + "OUTSDIF.d"
dimFolder = rootFolder + "include/SIF_Dimensions/Dimensions.csv"
figFolder = "/home/build/MT/figures"


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def load_SIF_params():

    Q = np.genfromtxt(pFolder + "Q.csv", delimiter=",")
    c = np.genfromtxt(pFolder + "c.csv", delimiter=",").reshape((-1, 1))
    A = np.genfromtxt(pFolder + "A.csv", delimiter=",")
    b = np.genfromtxt(pFolder + "b.csv", delimiter=",").reshape((-1, 1))
    D = np.genfromtxt(pFolder + "D.csv", delimiter=",")
    e = np.genfromtxt(pFolder + "e.csv", delimiter=",").reshape((-1, 1))
    x_lb = np.genfromtxt(pFolder + "x_lb.csv", delimiter=",")
    x_ub = np.genfromtxt(pFolder + "x_ub.csv", delimiter=",")
    x0 = np.genfromtxt(pFolder + "x0.csv", delimiter=",").reshape((-1, 1))
    x_traj = np.genfromtxt(pFolder + "x_iter.csv", delimiter=",")
    z_traj = np.genfromtxt(pFolder + "z_iter.csv", delimiter=",")
    return Q, c, A, b, D, e, x_lb, x_ub, x0, x_traj, z_traj


if __name__ == '__main__':

    with open(pFolder + 'probname.txt', 'r') as file:
        probname = file.read()
    mu = 1
    mu_list = np.genfromtxt(pFolder + "Barrier_Central_Path/mu.csv")
    dims = np.genfromtxt(dimFolder, delimiter=",", dtype=int)

    f = eval("load_SIF_{}_{}_{}(outsdif)".format(dims[0], dims[1], dims[2]))
    phi = eval("load_logbarrier_SIF_{}_{}_{}(f, mu)".format(
        dims[0], dims[1], dims[2]))
    # f = load_SIF_2_1_0(pFolder + "OUTSDIF.d")
    # phis = [load_barrier_SIF_2_1_0(f, m) for m in mus]
    # barrier_subproblem_solve(f, x0, mu)

    x_traj = np.genfromtxt(
        pFolder + "Barrier_Central_Path/x.csv", delimiter=", ")
    z_traj = np.genfromtxt(
        pFolder + "Barrier_Central_Path/z.csv", delimiter=", ")

    min_x = np.min(x_traj, axis=0)
    max_x = np.max(x_traj, axis=0)
    if mu_list.shape[0] > 9:
        mu_list = list(split(mu_list, 9))
    else:
        mu_list = [[x] for x in mu_list]

    plot_slack_x = (max_x-min_x)/10
    # x0 = np.linspace(min_x[0] - plot_slack_x[0], max_x[0] + plot_slack_x[0], 100)
    # x1 = np.linspace(min_x[1] - plot_slack_x[1], max_x[1] + plot_slack_x[1], 100)
    dx_prev = np.inf
    x_central = np.genfromtxt(pFolder + "Barrier_Central_Path/x.csv", delimiter =", ")
    N_mu =  len(mu_list)
    for i in range(x_central.shape[0]-1):
        if np.all(x_central[i+1,:] == x_central[i,:]):
            N_mu = i
            break
    mu_list = mu_list[:N_mu]
    
    
    Plot_name = "QP_Single_Constraint_Convergence"
    if (len(mu_list) == 8):
        fig, axes = plt.subplots(2, 4)
    elif (len(mu_list) > 3):
        fig, axes = plt.subplots(2, int(len(mu_list)/2))
    else:
        fig, axes = plt.subplots(1, int(len(mu_list)))

    axes = axes.ravel()

    for i, (mus, ax) in enumerate(zip(mu_list, axes)):
        xk_traj = x_central[i,:].reshape((-1,2))
        phi = eval(
            "load_logbarrier_SIF_{}_{}_{}(f, mus[-1])".format(dims[0], dims[1], dims[2]))
        for mu in mus:
            xk_traj = np.concatenate([xk_traj, np.genfromtxt(
                pFolder + "Subproblem_Iterations/mu_{m:.6f}/x.csv".format(m=mu), delimiter=", ").reshape((-1,2))])

        xmin = np.min(xk_traj, axis=0)
        xmax = np.max(xk_traj, axis=0)
        xplot_s = xmin - (xmax-xmin)/2
        xplot_f = xmax + (xmax-xmin)/2

        if xplot_s[0] == xplot_f[0]:
            xplot_s[0] = min(xplot_s)
            xplot_f[0] = max(xplot_f)
        if xplot_s[1] == xplot_f[1]:
            xplot_s[1] = min(xplot_s)
            xplot_f[1] = max(xplot_f)

        x0 = np.linspace(xplot_s[0], xplot_f[0], 100)
        x1 = np.linspace(xplot_s[1], xplot_f[1], 100)
        x0_m = np.linspace(xplot_s[0], xplot_f[0], 20)
        x1_m = np.linspace(xplot_s[1], xplot_f[1], 20)
        X_markers, Y_markers = np.meshgrid(x0_m, x1_m)

        X, Y = np.meshgrid(x0, x1)

        C = np.zeros(X.shape)
        Phi = np.zeros(X.shape)
        C_I = np.zeros((X.shape[0], X.shape[1], 2))
        grad_Phi = np.zeros(X.shape)
        x_ = list(zip(X.flatten(), Y.flatten()))
        C = np.reshape(list(map(lambda x: f(x), x_)), (X.shape))
        Phi = np.reshape(list(map(lambda x: phi(x), x_)), (X.shape))

        ax.scatter(X_markers, Y_markers, marker='x', s=2, color='k')
        ax.contourf(X, Y, Phi, cmap='Greys')
        # , cmap='Greys')#, levels=np.linspace(C.min().min(), C.max().max(), 10))
        CS = ax.contour(X, Y, C, cmap='Greys')
        ax.set_xlim(x0[0], x0[-1])
        ax.set_ylim(x1[0], x1[-1])
        # ax.contourf(X, Y, grad_Phi, levels = np.linspace(1e-6,1000,10), colors='k')
        # ax.contour(X, Y, C_I[:, :, 0], levels=[0], colors='k')
        # ax.contour(X, Y, C_I[:, :, 1], levels=[0], colors='k')
        # ax.contourf(X, Y, grad_Phi)
        ax.scatter(xk_traj[0, 0], xk_traj[0, 1], color='k', marker='x', s=.5)
        ax.scatter(xk_traj[-1, 0], xk_traj[-1, 1], color='k', marker='.')
        fig.suptitle('')

        from matplotlib.ticker import ScalarFormatter

        y_formatter = ScalarFormatter(useOffset=False)
        ax.yaxis.set_major_formatter(y_formatter)
        # ax.plot([xs[0], xf[0]], [xs[1], xf[1]], color='k')
        ax.plot(xk_traj[:, 0], xk_traj[:, 1], color='k')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'))

        title = '$\mu = $'
        for m in mus:
            title = title + str(m) + ", "
        ax.set_title(title)

    # plt.show()
    fig.subplots_adjust(hspace=1.)
    fig.subplots_adjust(wspace=1.)
    fig.savefig(figFolder + probname[:-5] + "_Trajectory.eps", format='eps')
