import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Some technical settings
# import matplotlib as mpl
np.set_printoptions(precision=3)
# np.random.seed(0)


NN = 4 # number of agents
n_l = 2 # number of leaders
d = 2 # dimension of positions and velocities

# followers model
# p_i'(t)=v_i(t)
# v_i'(t)=u_i(t)  for all i in V_f={n_l+1,...,NN}

# positions preallocation
# p_l=np.zeros((d*n_l,1))
L=2
p_l_star=np.array([[L,L,L,0]]).T
p_l=p_l_star
p_f=np.zeros((d*(NN-n_l),1))
# p_f=np.random.randn(d*(NN-n_l))
p_f=np.reshape(p_f,(d*(NN-n_l),1))
p_f_star=np.array([[0,0,0,L]]).T
p=np.concatenate((p_l,p_f))
p_star=np.concatenate((p_l_star,p_f_star))

# velocities
v_l=np.zeros((d*n_l,1))
v_f=np.random.randn(d*(NN-n_l))
v_f=np.reshape(v_f,(d*(NN-n_l),1))
v=np.concatenate((v_l,v_f))

# ER Network generation
p_ER = 0.6

I_NN = np.identity(NN, dtype=int)

while 1:
    G = nx.binomial_graph(NN,p_ER)
   
    Adj = nx.adjacency_matrix(G)
    Adj = Adj.toarray()

    test = np.linalg.matrix_power((I_NN+Adj),NN)

    if np.all(test>0):
        print("the graph is connected")
        break 
    else:
        print("the graph is NOT connected")
        
# nx.draw(G, with_labels = True)

# Bearing 
def bearing_matrix(p):
    # p_part=np.reshape(p,((NN,1,d)))
    GG=np.zeros((NN,NN,d))
    for ii in range(0,NN):
        for jj in range(0,NN):
            if ii != jj:
                a=(p[d*jj:d*(jj+1)]-p[d*ii:d*(ii+1)])/(np.linalg.norm(p[d*jj:d*(jj+1)]-p[d*ii:d*(ii+1)]))
                GG[ii,jj]=a.T

    return GG

# target formation
GG_star=bearing_matrix(p_star)
print(GG_star)

# orthogonal projection matrix 
def ortproj(g_ij):
    P_g_ij=np.eye(d)-g_ij@g_ij.T
    return P_g_ij

b=np.array([GG_star[0,1]]).T
P_g_01_star=ortproj(b)
print(P_g_01_star)

# formation maneuvering with constant leader velocity
def maneuvering(p,v,k_p,k_v,P_g_ij_star,ii,jj):
    # u=0
    u=np.zeros((d,1))
    # p=np.array(p).T
    # v=np.array(v).T
    # for ii in range (n_l,NN):
#    N_ii = np.nonzero(Adj[ii])[0] # In-Neighbors of node i
#    for jj in range (N_ii):
    a=k_p*(p[d*ii:d*(ii+1)]-p[d*jj:d*(jj+1)])
    a=np.reshape(a,(d,1))
    b=k_v*(v[d*ii:d*(ii+1)]-v[d*jj:d*(jj+1)])
    b=np.reshape(b,(d,1))
    c=a+b
    u=P_g_ij_star@c
    u=-u
    # u=np.array(u).T
    return u

# gain coefficients
k_p=1
k_v=1

dt=1e-3 # integration step
horizon=10 # [s]
# preallocation
T=int(horizon/dt)
pp=np.zeros((d*NN,T))
# pp[:,0]=np.vstack(p)
# pp[:,]=np.reshape(p,(d*NN,1))
pp[:,]=np.reshape(p_star,(d*NN,1))
vv=np.zeros((d*NN,T))
# vv[:,]=np.reshape(v,(d*NN,1))
aa=np.zeros((d*NN,T))
for tt in range(0,T):
    if tt>=1:
        for ii in range(n_l,NN):
             N_ii = np.nonzero(Adj[ii])[0] # In-Neighbors of node i
             u_i=np.zeros((d,1))
             for jj in N_ii:
                b=np.array([GG_star[ii,jj]]).T
                P_g_ij_star=ortproj(b)  
                u_i+=maneuvering(pp[:,tt-1],vv[:,tt-1],k_p,k_v,P_g_ij_star,ii,jj)
                
             aa[d*ii:d*(ii+1),tt]=np.array(u_i).T # [m/s^2]
             vv[d*ii:d*(ii+1),tt]=aa[d*ii:d*(ii+1),tt]*dt+vv[d*ii:d*(ii+1),tt-1]  # [m/s]
             pp[d*ii:d*(ii+1),tt]=vv[d*ii:d*(ii+1),tt]*dt+pp[d*ii:d*(ii+1),tt-1] # [m]

             # p=np.reshape((pp[:,tt]),(d*NN,1))
             # v=np.reshape((vv[:,tt]),(d*NN,1))
           
print(pp)

# figures
X=[L,L,pp[4,0],pp[6,0]]
Y=[L,0,pp[5,0],pp[7,0]]
plt.plot(X,Y, marker="o", color="green", linestyle="")
plt.xlabel("$x$ [m]")
plt.ylabel("$y$ [m]")
plt.title("Initial configuration")
plt.grid()
plt.show()

# plt.figure()
X=[L,L,pp[4,T-1],pp[6,T-1]]
Y=[L,0,pp[5,T-1],pp[7,T-1]]
plt.plot(X,Y, marker="o", color="red", linestyle="")
plt.xlabel("$x$ [m]")
plt.ylabel("$y$ [m]")
plt.title("Final configuration")
plt.grid()
plt.show()


# Local Estimates
t=np.linspace(0,horizon,T)
for ii in range(n_l):
	plt.plot(t, pp[d*ii,:],label="$x_l$")

for ii in range(n_l):
    plt.plot(t,pp[d*(ii+1),:],label="$y_l$")

for ii in range(n_l,NN):
    plt.plot(t,pp[d*ii,:],label="$x_f$")

for ii in range(n_l,NN):
    plt.plot(t,pp[d*(ii)+1,:],label="$y_f$")

plt.legend(loc="lower right", title="Legend Title", frameon=False)
#df = pd.DataFrame({"Leaders": [pp[0,:],pp[2,:]],
                    #"Followers" : [pp[4,:],pp[6,:]]})
#plt.plot(df)
#legend = plt.legend(['Leaders','Followers'], title = "Legend")

plt.xlabel("iterations $t$ [s]")
plt.ylabel("$p_{i,t}$")
plt.title("Evolution of the state")
plt.grid()
plt.show()

""""
for ii in range(NN):    
    plt.plot(t,pp[d*(ii+1),:])

plt.xlabel("iterations $t$ [s]")
plt.ylabel("$x_{i,t}$")
plt.title("Evolution of the state")

plt.grid()
plt.show()"""
