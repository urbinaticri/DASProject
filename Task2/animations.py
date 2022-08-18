#
# Formation Control Algorithm
# Lorenzo Pichierri, Andrea Testa
# Bologna, 14/02/2022
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def formation(xx, horizon, Adj, NN, n_x,animate=True):

    TT = np.size(horizon,0)
    for tt in range(np.size(horizon,0)):
        xx_tt = xx[:,tt].T
        for ii in range(NN):
            for jj in range(NN):
                index_ii =  ii*n_x + np.array(range(n_x))
                p_prev = xx_tt[index_ii]
                plt.plot(p_prev[0],p_prev[1], marker='o', markersize=20, fillstyle='none')
                if Adj[ii,jj]>1 and (jj>ii):
                    index_jj = (jj % NN)*n_x + np.array(range(n_x))
                    p_curr = xx_tt[index_jj]
                    plt.plot([p_prev[0],p_curr[0]],[p_prev[1],p_curr[1]], 
                                linewidth = 2, color = 'tab:blue', linestyle='solid')

        axes_lim = (np.min(xx)-1,np.max(xx)+1)
        plt.xlim(axes_lim)
        plt.ylim(axes_lim)
        plt.plot(xx[0:n_x*NN:n_x,:].T,xx[1:n_x*NN:n_x,:].T)
        plt.axis('equal')

        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()