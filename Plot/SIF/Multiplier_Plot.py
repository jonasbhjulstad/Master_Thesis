
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys
import os
from os.path import basename
from os import path
matplotlib.use
sys.path.append("//home/build/FIPOPT/build/test/Plot/")
baseFolder = "//home/build/FIPOPT/Data/SIF/"
rootFolder = "//home/build/FIPOPT/"
figFolder = "//home/deb/Documents/MT/figures/"


def read_z_traj(objective_path):
    mus = np.genfromtxt(objective_path + "mu.csv")
    z_traj = []
    for mu in mus:
        if path.exists(objective_path + "mu_{m:.6f}/".format(m=mu)):
            z_traj.append(np.genfromtxt(
                    objective_path + "mu_{m:.6f}/z.csv".format(m=mu), delimiter=", "))
            if z_traj[-1].ndim == 1:
                z_traj[-1] = z_traj[-1].reshape((1,-1))
    return z_traj

def Plot_Multipliers(fname):

    eq_dirs = []
    ineq_dirs = []

    for filename in os.listdir(fname):
        if filename.startswith("Inequality_Restoration"): 
            ineq_dirs.append(filename + "/")
        elif filename.startswith("Equality_Restoration"): 
            eq_dirs.append(filename + "/")

    z_vals = read_z_traj(fname)
    fig, ax = plt.subplots()
    a = 0
    if z_vals != []:
        for k, zk in enumerate(z_vals):
            if zk.size == 0:
                z_vals[k] = np.zeros(z_vals[-1].shape)
            ax.scatter(np.full(1, a), zk[0,0], color='k', marker='.')
            # ax.scatter(k, zk[0,1], color='k', marker='.')
            a += zk.shape[0]

        z_vals = np.concatenate(z_vals, axis=0)
        Ng = len(z_vals[0])-4
        if Ng > 0:
            ax.plot(z_vals[:,:Ng], color='k')
            ax.plot(z_vals[:,0], color='k', label=r'$z_{g,k}^{(j)}$')
            ax.plot(z_vals[:, Ng:Ng+2], color='k', linestyle='dotted')
        ax.plot(z_vals[:, :-2], color='k', linestyle='dashed')
        ax.plot(z_vals[:, Ng], color='k', linestyle='dotted', label=r'$z_{ub, k}^{(j)}$')
        ax.plot(z_vals[:, -1], color='k', linestyle='dashed', label=r'$z_{lb, k}^{(j)}$')
        ax.plot(z_vals[:, :-2], color='k', linestyle='dashed')
        
        # ax.plot(z_vals[:,1], color='k', linestyle='dotted', label=r'$z_{lb/ub,k}^{(j)}$')
        # ax.plot(z_vals[:,2], color='k', linestyle='dashed', label=r'$\varphi_{\mu_j}(x_k^{(j)})$')
        ax.grid()
        # ax.set_yscale('log')
        ax.set_xlabel(r'Iterations $k$ for subproblems $j$')

        ax.legend()

        fig.subplots_adjust(hspace=1.)
        fig.subplots_adjust(wspace=1.)
        # plt.show()
        fig.savefig(figFolder + basename(fname[:-1])[:-4] + "_Multipliers_linear.pdf")
        plt.close()

if __name__ == '__main__':


    if len(sys.argv) > 1:
        Plot_Multipliers(baseFolder + sys.argv[1] + "/")
    else:
        with open(baseFolder + 'Problem/probname.txt', 'r') as file:
            probname = file.read()[:-1]
        print(probname)
        Plot_Multipliers(baseFolder + probname + "/")

