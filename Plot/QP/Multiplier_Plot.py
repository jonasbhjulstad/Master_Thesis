

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import ticker as mticker
import sys
import os
matplotlib.use
sys.path.append("/home/deb/Documents/FIPOPT/build/test/Plot/")

baseFolder = "/home/deb/Documents/FIPOPT/Data/QP/"
pFolder = baseFolder + "Param/"
dFolder = baseFolder + "Trajectory/"




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
    A = np.genfromtxt(pFolder + "A.csv", delimiter=",").reshape((-1,Nx))
    b = np.genfromtxt(pFolder + "b.csv", delimiter=",").reshape((-1,1))
    D = np.genfromtxt(pFolder + "D.csv", delimiter=",").reshape((-1,Nx))
    e = np.genfromtxt(pFolder + "e.csv", delimiter=",").reshape((-1,1))
    x_lb = np.genfromtxt(pFolder + "x_lb.csv", delimiter=",")
    x_ub = np.genfromtxt(pFolder + "x_ub.csv", delimiter=",")
    x0 = np.genfromtxt(pFolder + "x0.csv", delimiter=",").reshape((-1,1))
    # x_traj = np.genfromtxt(pFolder + "x_iter.csv", delimiter=",")
    # z_traj = np.genfromtxt(pFolder + "z_iter.csv", delimiter=",")
    return Q, fix_dim_vec(c), fix_dim_mat(A), fix_dim_vec(b), fix_dim_mat(D), fix_dim_vec(e), x_lb, x_ub, x0



rootFolder = "/home/deb/Documents/FIPOPT/"
sys.path.append(rootFolder + "build/test/Plot/")

figFolder = "/home/deb/Documents/MT/figures/"


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def read_x_traj(objective_path):
    mus = np.genfromtxt(objective_path + "mu.csv")
    xk_traj = []
    for mu in mus:
        xk_traj.append(np.genfromtxt(
                objective_path + "mu_{m:.6f}/x.csv".format(m=mu), delimiter=", ")[:2].reshape((-1,2)))
    return xk_traj

def read_z_traj(objective_path):
    mus = np.genfromtxt(objective_path + "mu.csv")
    z_traj = []
    for mu in mus:
        z_traj.append(np.genfromtxt(
                objective_path + "mu_{m:.6f}/z.csv".format(m=mu), delimiter=", "))
        if z_traj[-1].ndim == 1:
            z_traj[-1] = z_traj[-1].reshape((1,-1))
    return z_traj

if __name__ == '__main__':

    mu = 1
    mu_list = np.genfromtxt(dFolder + "mu.csv")

    objvals = np.genfromtxt(dFolder + "obj.csv", delimiter=", ");

    eq_dirs = []
    ineq_dirs = []

    for filename in os.listdir(dFolder):
        if filename.startswith("Inequality_Restoration"): 
            ineq_dirs.append(filename + "/")
        elif filename.startswith("Equality_Restoration"): 
            eq_dirs.append(filename + "/")

    z_vals = read_z_traj(dFolder)

    fig, ax = plt.subplots()
    a = 0

    for k, zk in enumerate(z_vals):
        if zk.size == 0:
            z_vals[k] = np.zeros(z_vals[-1].shape)
        ax.scatter(np.full(1, a), zk[0,0], color='k', marker='.')
        # ax.scatter(k, zk[0,1], color='k', marker='.')
        a += zk.shape[0]

    z_vals = np.concatenate(z_vals, axis=0)
    ax.plot(z_vals[:,0], color='k', label=r'$z_{g,k}^{(j)}$')
    # ax.plot(z_vals[:,1], color='k', linestyle='dotted', label=r'$z_{lb/ub,k}^{(j)}$')
    # ax.plot(z_vals[:,2], color='k', linestyle='dashed', label=r'$\varphi_{\mu_j}(x_k^{(j)})$')
    ax.grid()
    ax.set_yscale('log')
    ax.set_xlabel(r'Iterations $k$ for subproblems $j$')

    ax.legend()
    # ax.plot(z_vals[:,2])
    # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'))

        # title = '$\mu = $'
        # for m in mus:
        #     title = title + str(m) + ", "
    # ax.set_title("$x^* = [{:.1f}, {:.1f}]$, $f^* = {:.3e}$, $E_0 = {:.3e}$, $N_\mu = {}$".format(x_cp_traj[-1][0], x_cp_traj[-1][1], objvals[1], objvals[0], len(mu_list)))
    # plt.show()
    fig.subplots_adjust(hspace=1.)
    fig.subplots_adjust(wspace=1.)
    # plt.show()
    # fig.savefig(figFolder + "NONSOC_Multipliers.eps", format='eps')
    fig.savefig(figFolder + "NC_QP_Multipliers.eps", format='eps')
