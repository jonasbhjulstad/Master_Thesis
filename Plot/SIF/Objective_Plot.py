
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import os
from os.path import basename
from os import path

sys.path.append("/home/deb/Documents/gitFIPOPT/Release/Plot/SIF/")
sys.path.append("/home/deb/Documents/gitFIPOPT/Plot/SIF")
baseFolder = "/home/deb/Documents/gitFIPOPT/Data/SIF"
HS_Folder = baseFolder + "/HS"
is_restoration_plot = True

rootFolder = "/home/deb/Documents/gitFIPOPT/"
figFolder = "/home/deb/Documents/gitFIPOPT/figures/"
sys.path.append(rootFolder + "build/test/Plot/")

def read_f_obj(objective_path):
    mu_dirs = []
    f_obj = []
    for dirname in os.listdir(objective_path):
        if 'mu_' in dirname:
            mu_dirs.append(dirname)
    mu_dirs.sort(key=lambda x: float(x.split('_')[-1]))
    for dirname in reversed(mu_dirs):
        f_obj.append(np.genfromtxt(
            objective_path + dirname + "/obj.csv", delimiter=", ").reshape((-1,2)))
    return f_obj

def read_thetas(theta_path):
    mus = np.genfromtxt(theta_path + "mu.csv").reshape((-1))
    theta = []
    for mu in mus:
        if path.exists(theta_path + "mu_{m:.6f}".format(m=mu)):
            theta.append(np.genfromtxt(
                theta_path + "mu_{m:.6f}/theta.csv".format(m=mu)).reshape((-1,1)))
    return theta
def fill_background(ax, inds):
    for (x0, x1) in inds:
        rect = patches.Rectangle((x0, ax.get_ylim()[0]), x1 - x0, ax.get_ylim()[1], facecolor='gray', alpha=.3)
        ax.add_patch(rect)

def Plot_Objective(fname):

    print(fname)
    f_vals = read_f_obj(fname)
    theta_vals = read_thetas(fname)

    ineq_dirs, eq_dirs = [], []
    ineq_inds, eq_inds = [], []
    ineq_f_vals, eq_f_vals = [], []
    ineq_th_vals, eq_th_vals = [], []
    for filename in os.listdir(fname):
        if filename.startswith("Inequality_Restoration"):
            ineq_inds.append(int(filename[-1])) 
            ineq_dirs.append(filename + "/")
            ineq_inds.append(int(filename[-1]))
            ineq_f_vals.append(read_f_obj(fname + filename + "/"))
            ineq_th_vals.append(read_thetas(fname + filename + "/"))
        elif filename.startswith("Equality_Restoration"):
            eq_inds.append(int(filename[-1])) 
            eq_dirs.append(filename + "/")
            eq_inds.append(int(filename[-1]))
            eq_f_vals.append(read_f_obj(fname + filename + "/"))
            eq_th_vals.append(read_thetas(fname + filename + "/"))

    ylim = [[np.inf, np.inf], [-np.inf, -np.inf]]
    is_restoration = ineq_inds or eq_inds


    if f_vals:
        box_inds = []
        is_constrained = not np.all(theta_vals[0] == 0)
        if is_constrained:
            fig, ax = plt.subplots(3)
        else:
            fig, ax = plt.subplots(2)

        ax_res = [x.twinx() for x in ax]
        k = 0
        limFac = 1.2
        # ax[0].set_ylim(ylim[0][1]/limFac, ylim[1][1]*limFac)
        # ax[1].set_ylim(ylim[0][0]/limFac, ylim[1][0]*limFac)
        # if is_restoration:
        #     ax_res[0].set_ylim(ylim_res[0][1]/limFac, ylim_res[1][1]*limFac)
        #     ax_res[1].set_ylim(ylim_res[0][0]/limFac, ylim_res[1][0]*limFac)
        #     ax_res[2].set_ylim(ylim_res[0][0]/limFac, ylim_res[1][0]*limFac)
        for i, (obj, th) in enumerate(zip(f_vals, theta_vals)):
            res_f = []
            if i in ineq_inds:
                res_f = ineq_f_vals.pop(0)
                if is_constrained:
                    res_th = ineq_th_vals.pop(0)
            elif i in eq_inds:
                res_f = eq_f_vals.pop(0)
                if is_constrained:
                    res_th = eq_th_vals.pop(0)
            
            if np.any(res_f) and is_restoration_plot:
                if is_constrained:
                    for res_theta in res_th:
                        xt = list(range(k, k+ res_theta.shape[0], 1))
                        ax_res[2].scatter([xt[0], xt[-1]], [res_theta[0], res_theta[-1]], color='k', marker='.')
                        ax_res[2].plot(xt, res_theta, color='k', label=r"$\theta_R(x_k^{(j)})$")        
                for res_obj in res_f:
                    xt = list(range(k, k+ res_obj.shape[0], 1))
                    box_inds.append((xt[0], xt[-1]))
                    ax_res[0].plot(xt, res_obj[:,1], color='k', label=r"$f_R(x_k^{(j)})$")
                    ax_res[1].plot(xt, res_obj[:,0], color='k', label=r"$E_{R,\mu_j}(x_k^{(j)})$")        
                    ax_res[0].scatter([xt[0], xt[-1]], [res_obj[0,1], res_obj[-1, 1]], color='k', marker='.')
                    ax_res[1].scatter([xt[0], xt[-1]], [res_obj[0,0], res_obj[-1, 0]], color='k', marker='.')
                    k += res_obj.shape[0]-1
                    obj_prev = []
                    th_prev = []

            if (i != 0) and (np.any(obj_prev)):
                ax[0].plot([k-1, k], [obj_prev[1], obj[0,1]], color='k')
                ax[1].plot([k-1, k], [obj_prev[0], obj[0,0]], color='k')
            if is_constrained and (i != 0) and (np.any(th_prev)):
                ax[2].plot([k-1, k], [th_prev, th[0]], color='k', label=r"$\theta(x_k^{(j)})$")        
            xt = list(range(k, k+ obj.shape[0], 1))
            ax[0].plot(xt, obj[:,1], color='k', label=r"$f(x_k^{(j)})$")
            ax[1].plot(xt, obj[:,0], color='k', label=r"$E_{\mu_j}(x_k^{(j)})$")        
            ax[0].scatter([xt[0], xt[-1]], [obj[0,1], obj[-1, 1]], color='k', marker='.')
            ax[1].scatter([xt[0], xt[-1]], [obj[0,0], obj[-1, 0]], color='k', marker='.')
            k += obj.shape[0]
            if is_constrained and len(th) > 1:
                ax[2].scatter([xt[0], xt[1]], [th[0], th[-1]], color='k', marker='.')
                ax[2].plot(xt, th, color='k', label=r"$\theta(x_k^{(j)})$")  
                th_prev = th[-1]      

            obj_prev = obj[-1,:]
        

        # for ax_k in ax:
        #     handles, labels = ax_k.get_legend_handles_labels()
        #     handle_list, label_list = [], []
        #     for handle, label in zip(handles, labels):
        #         if label not in label_list:
        #             handle_list.append(handle)
        #             label_list.append(label)
        #     ax_k.legend(handle_list, label_list)
        ax[0].set_title(r'Objective $f$')

        ax[1].set_title(r'Barrier Optimality $E_{\mu_j}$')
        if is_constrained:
            ax[2].set_title(r'Constraint Violation $\theta$')
        f_vals = np.concatenate(f_vals, axis=0)
        theta_vals = np.concatenate(theta_vals, axis=0)
        for x in ax:
            x.grid()
        ax[-1].set_xlabel(r'Iterations $k$')
        if is_restoration:
            _ = [x.set_ylabel('Restoration') for x in ax_res]
        fig.subplots_adjust(hspace=1.)
        fig.subplots_adjust(wspace=1.)
        fig.savefig(figFolder + basename(fname[:-1])[:-4] + "_Objective_Linear.pdf")
        [x.locator_params(axis="y", nbins=3) for x in ax_res]
        [fill_background(ax_k, box_inds) for ax_k in ax_res]

        # plt.show()
        _  = [x.set_yscale('symlog') for x in ax]
        _  = [x.set_yscale('symlog') for x in ax_res]
        [x.locator_params(axis="y", numticks=5) for x in ax_res]

        # plt.show()
        fig.savefig(figFolder + basename(fname[:-1])[:-4] + "_Objective.pdf")
        # plt.close()

if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        Plot_Objective(HS_Folder + sys.argv[1] + "/")
    else:
        with open(baseFolder + '/Problem/probname.txt', 'r') as file:
            probname = file.read()[:-1]
        Plot_Objective(baseFolder + "/" + probname + "/")
        # Plot_Objective(HS_Folder + "/" + "HS6.SIF" + "/")


