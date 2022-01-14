import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import ticker as mticker
import sys
import os
from os.path import basename

matplotlib.use
sys.path.append("/home/deb/Documents/FIPOPT/Release/Plot/SIF/")

SIF_Folder = "/home/deb/Documents/FIPOPT/Data/SIF/"
HS_Folder = SIF_Folder + "HS/"
pFolder = SIF_Folder + "Problem/"
dimFolder = "/home/deb/Documents/FIPOPT/include/SIF_Dimensions/Dimensions.csv"
from Binder_SIF import *


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

rootFolder = "/home/deb/Documents/FIPOPT/"
sys.path.append(rootFolder + "build/test/Plot/")

figFolder = "/home/deb/Documents/FIPOPT/figures/"
outsdif = pFolder + "OUTSDIF.d" 


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def read_x_traj(objective_path, restoration_iters = []):
    mus = np.genfromtxt(objective_path + "mu.csv").reshape((-1))
    xk_traj = []
    for mu in mus:
        xmu = np.genfromtxt(objective_path + "mu_{m:.6f}/x.csv".format(m=mu), delimiter=", ")
        if len(xmu.shape) > 1:
            xmu = xmu[:,:2].reshape((-1,2))
        else:
            xmu = xmu[:2].reshape((-1,2))
        xk_traj.append(xmu)
    # x_sliced = [xk_traj[0]]
    # for i, xk in enumerate(xk_traj[1:]):
    #     if i in restoration_iters:
    #         x_sliced.append([])
    #     x_sliced[-1] = np.concatenate([x_sliced[-1], xk], axis=0)
    return xk_traj

def get_dim(fname):
    with open(fname, 'r') as file:
        res = file.readline()
        if (res == '\n'):
            return 0
        else:
            return len(np.fromstring(res, sep=', '))
def is_converged(dirname):
    with open(HS_Folder + dirname + "/success.txt", 'r') as file:
        if int(file.readline()) == 1:
            return True

def Plot_Trajectory(fname):

    Nx = get_dim(fname + "x.csv")
    Ng = get_dim(fname + "z.csv") - 2*Nx
    Nh = get_dim(fname + "lbd.csv")

    if Nx != 2:
        return
    
    with open('SIF_traj_log.txt', 'a') as file:
        file.write(basename(fname[:-1]) + "\n")
    
    mu_list = np.genfromtxt(fname + "mu.csv")

    objvals = np.genfromtxt(fname + "obj.csv", delimiter=", ");
    mu = 0.01
    f = eval("load_SIF_{}_{}_{}(outsdif)".format(Nx, Ng, Nh))
    phi = eval("load_barrier_SIF_{}_{}_{}(f, mu)".format(
        Nx, Ng, Nh))
    fig, ax = plt.subplots()


    x_cp_traj = np.genfromtxt(fname + "x.csv", delimiter=",")
    # if not is_converged(basename(fname[:-1])):
    #     smallmu = np.inf
    #     for filename in os.listdir(fname):
    #         if filename.startswith('mu_'):
    #             smallmu = np.min([(float(filename.split('_')[-1])), smallmu])
    #             with open(fname + "mu_{:.6f}/x.csv".format(smallmu), 'r') as file:
    #                 x_end = np.fromstring(file.readlines()[-1], sep=', ').reshape((-1,2))
    #                 x_cp_traj = np.concatenate([x_cp_traj, x_end], axis=0)
    xk_traj = x_cp_traj[0,:2].reshape((1,2))
    xmax = np.full(xk_traj.shape, -np.inf)
    xmin = -xmax
    restoration_iters = []
    ineq_dirs, eq_dirs = [], []
    ineq_inds, eq_inds = [], []
    for filename in os.listdir(fname):
        if filename.startswith("Inequality_Restoration"): 
            ineq_dirs.append(filename + "/")
            ineq_inds.append(int(filename[-1]))
            restoration_iters.append(filename[-1]) 
        elif filename.startswith("Equality_Restoration"): 
            eq_dirs.append(filename + "/")
            eq_inds.append(int(filename[-1]))
            restoration_iters.append(filename[-1])


    for idir in ineq_dirs:
        # cI_x = np.genfromtxt(fname + idir + "x.csv", delimiter=",")
        cI_x = np.concatenate(read_x_traj(fname + idir), axis=0)
        if len(cI_x.shape) > 1:
            cI_x = cI_x[:,:2]
        else:
            # cI_x = cI_x[:2].reshape((1,-1))
            break
        xmin = np.min(np.concatenate([xmin, cI_x], axis=0), axis=0).reshape((1,-1))
        xmax = np.max(np.concatenate([xmax, cI_x], axis=0), axis=0).reshape((1,-1))
        # ax.scatter(cI_x[0,0], cI_x[0,1], marker='.', color='k')
        # ax.scatter(cI_x[-1,0], cI_x[-1,1], marker='.', color='k')
        ax.scatter([cI_x[0,0], cI_x[-1,0]], [cI_x[0,1], cI_x[-1,1]], color='k', marker='P');
        ax.plot(cI_x[:,0], cI_x[:,1], color='k', linestyle='dotted', alpha=.5, label='Inequality Restoration');
    for edir in eq_dirs:
        cE_x = np.genfromtxt(fname + edir + "x.csv", delimiter=",")
        if len(cE_x.shape) > 1:
            cE_x = cE_x[:,:2]
        else:
            # cE_x = cE_x[:2].reshape((1,-1))
            break
        


        xmin = np.min(np.concatenate([xmin, cE_x], axis=0), axis=0).reshape((1,-1))
        xmax = np.max(np.concatenate([xmax, cE_x], axis=0), axis=0).reshape((1,-1))
        # ax.scatter(cE_x[0,0], cE_x[0,1], marker='.', color='k')
        # ax.scatter(cE_x[-1,0], cE_x[-1,1], marker='.', color='k')
        ax.scatter([cE_x[0,0], cE_x[-1,0]], [cE_x[0,1], cE_x[-1,1]], color='k', marker='*', label='Inequality');
        ax.plot(cE_x[:,0], cE_x[:,1], linestyle='dashed', color='k', alpha=.5, label='Equality Restoration');

    main_x_traj = read_x_traj(fname, restoration_iters)
    xmin = np.min(np.concatenate([xmin, np.concatenate(main_x_traj, axis=0)], axis=0), axis=0)
    xmax = np.max(np.concatenate([xmax, np.concatenate(main_x_traj, axis=0)], axis=0), axis=0)



    phi = eval(
        "load_barrier_SIF_{}_{}_{}(f, mu_list[-1])".format(Nx, Ng, Nh))
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
    CS = ax.contour(X, Y, C, cmap='Greys')
            

    Phi = np.reshape(list(map(lambda x: phi(x), x_)), (X.shape))

    ax.scatter(X_markers, Y_markers, marker='x', s=2, color='k')
    if Nh > 0:
        C_E = np.reshape(list(map(lambda x: np.linalg.norm(f.Eval_h(x)), x_)), (X.shape))
        [eCol, eRow] = np.where(np.abs(C_E) <= 1e-1)
        # plt.scatter(X[eCol, eRow], Y[eCol, eRow], marker='o', color='k', label=r'$c_E(x) = 0$')
        
        ax.contour(X, Y, C_E, levels=[-1e-1, 1e-1], cmap='Greys', linestyle='-o-')
    # res = ax.contour(X, Y, C_E, levels=[-1e-2, 1e-2], cmap='Greys')

    ax.contourf(X, Y, Phi, cmap='Greys')
    ax.set_xlim(xplot_s[0], xplot_f[0])
    ax.set_ylim(xplot_s[1], xplot_f[1])
    ax.scatter(xk_traj[0, 0], xk_traj[0, 1], color='k', marker='x', s=.5)
    ax.scatter(xk_traj[-1, 0], xk_traj[-1, 1], color='k', marker='.')


    from matplotlib.ticker import ScalarFormatter

    y_formatter = ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)

    # ax.scatter(x_cp_traj[0,0], x_cp_traj[0,1], color='k', marker='.')
    # ax.plot(x_ineq_cp_traj[:,0], x_ineq_cp_traj[:,1], color='k', linestyle='dotted', label='Restoration')
    # ax.scatter(x_cp_traj[1,0], x_cp_traj[1,0], color='k', marker='.')
    main_x_traj = np.concatenate(main_x_traj, axis=0)
    ax.plot(main_x_traj[:,0], main_x_traj[:,1], color='k', label='Barrier Subproblem')
    ax.scatter(x_cp_traj[:-1, 0], x_cp_traj[:-1,1], color='k', marker='.', label='Central Path')
    ax.scatter(main_x_traj[-1,0], main_x_traj[-1,1], color='k', marker='^', label=r"$x^*$")

    ax.set_title(r"$x^* = [{:.1f}, {:.1f}]$, $f^* = {:.3e}$, $E_0 = {:.3e}$, $N_\mu = {}$".format(x_cp_traj[-1][0], x_cp_traj[-1][1], objvals[-1,1], objvals[-1,0], len(mu_list)))
    fig.subplots_adjust(hspace=1.)
    fig.subplots_adjust(wspace=1.)
    ax.legend()
    plt.show()
    fig.savefig(figFolder + "_" + basename(fname[:-1])[:-4] + "_Trajectory.pdf")
    # plt.show()
    # fig.close()

if __name__ == '__main__':

    with open(SIF_Folder + 'Problem/probname.txt', 'r') as file:
        probname = file.read()[:-1]

    Plot_Trajectory(HS_Folder + probname + "/")


