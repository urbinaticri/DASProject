from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import control
import networkx as nx
from animations import containment as animation
from scipy.integrate import solve_ivp

np.random.seed(5)

NN = 4 # number of agents
n_leaders = 2 # number of leaders
d = 2 # dimension of positions and velocities

# positions
p = np.vstack((
	np.zeros((d*n_leaders,1)),
	np.zeros((d*(NN-n_leaders),1))
)) + 5*np.random.rand(d*NN,1)

# velocities
v = np.vstack((
	np.zeros((d*n_leaders,1)),
	np.zeros((d*(NN-n_leaders),1))
))

# bearing unit vector g_{ij}
def g(i,j):
	return (p[j,j+d] - p[i:i+d]) / (np.linalg.norm(p[j,j+d] - p[i:i+d])**2)

# orthogonal projection matrix P_{g_{ij}}
def P(g_ij):
	return np.identity(d) - g_ij@(g_ij.T)

# formation: square ex. in fig 2 -> agent 1 bottom-left, order counter-clockwise
L = 1
D =sqrt(1)
g_star = [[	[0,0],		[0,L],		[D,D],		[L,0]],
	 	  [	[0,L],		[0,0],		[L,0],		[D,D]],
	 	  [	[D,D],		[L,0],		[0,0], 		[0,L]],
	 	  [	[L,0],		[D,D],		[0,L],		[0,0]]]
g_star = np.array(g_star)
Pg_star = np.zeros((d*NN, d*NN))
for ii in range(NN):
	for jj in range(NN):
		Pg_star[ii:ii+d, jj:jj+d] = P(g_star[ii, jj])


p_ER = 0.9

I_NN = np.identity(NN, dtype=int)
I_nx = np.identity(d, dtype=int)
I_NN_nx = np.identity(d*NN, dtype=int)
O_NN = np.ones((NN,1), dtype=int)
	
# ER Network generation
while 1:
	graph_ER = nx.binomial_graph(NN,p_ER)
	Adj = nx.adjacency_matrix(graph_ER).toarray()
	# test connectivity
	test = np.linalg.matrix_power((I_NN+Adj),NN)
	
	if np.all(test>0):
		print("the graph is connected\n")
		break

DEGREE = np.sum(Adj, axis=0)
D_IN = np.diag(DEGREE)
# Laplacian Matrix
L_IN = D_IN - Adj.T

# Bearing Laplacian Matrix
B = np.zeros((d*NN, d*NN))
for ii in range(NN):
	N_ii = np.nonzero(Adj[ii])[0] # In-Neighbors of node i

	sumPg_ik_star = 0
	for jj in N_ii:
		B[ii:ii+d, jj:jj+d] = -Pg_star[ii:ii+d, jj:jj+d]
		sumPg_ik_star += Pg_star[ii:ii+d, jj:jj+d]
	B[ii:ii+d, ii:ii+d] = sumPg_ik_star

# Partitioning B
n_l = n_leaders
n_f = NN-n_leaders

B_ll = B[d*0:d*n_l, d*0:d*n_l]	# shape (d*n_l, d*n_l)
B_lf = B[d*0:d*n_l, d*n_f:d*NN]	# shape (d*n_l, d*n_f)
B_fl = B[d*n_f:d*NN,d*0:d*n_l]	# shape (d*n_f, d*n_l)
B_ff = B[d*n_f:d*NN,d*n_f:d*NN] # shape (d*n_f, d*n_f) -> if nonsingular then target formation is unique

BB_ext_up = np.concatenate((B_ll, B_lf), axis = 1)
BB_ext_low  = np.concatenate((B_fl, B_ff), axis = 1)
BB = np.concatenate((BB_ext_up, BB_ext_low), axis = 0)

# system dynamics: Formation Maneuvering with Constant Leader Velocity
def form_maneuv_clv_func(p, v, k_p, k_v, Adj):
	u = np.zeros(np.size(v))
	print(u)
	for ii in range(NN):
		N_ii = np.where(Adj[:,ii]>0)[0]
		index_ii =  ii*d + np.arange(d)*d
		for jj in N_ii:
			index_jj = jj*d + np.arange(d)*d
			pp = p[index_ii] - p[index_jj]
			vv = v[index_ii] - v[index_jj]
			u = P(g_star[index_ii, index_jj]) * (k_p*pp + k_v*vv)
	return u

L_f = L_IN[0:NN-n_leaders, 0:NN-n_leaders]
L_fl = L_IN[0:NN-n_leaders, NN-n_leaders:]
LL = np.concatenate((L_f, L_fl), axis = 1)
LL = np.concatenate((np.zeros((n_leaders,NN)), LL), axis = 0)

# replicate for each dimension
LL_kron = np.kron(LL,I_nx)


x_init = np.vstack((
	p,
	v
))

################
## followers ext with integral Action

k_i = 0.4
K_I = -k_i*I_NN_nx

LL_ext_up = np.concatenate((LL_kron, K_I), axis = 1)
LL_ext_low = np.concatenate((np.zeros(LL_kron.shape), LL_kron), axis = 1)
LL_ext = np.concatenate((LL_ext_up, LL_ext_low), axis = 0)

BB = np.concatenate((np.zeros((NN*d,NN*d)), BB), axis = 0)

A = -LL_ext
B = BB
C = np.identity(np.size(LL_ext,axis = 0)) # to comply with StateSpace syntax

################
sys = control.StateSpace(A,B,C,0) # dx(p,v) = -L x(p,v) + B u

dt = 0.01
Tmax = 10.0
horizon = np.arange(0.0, Tmax, dt)

k_p = 0.5
k_v = 0.5

x_out = np.zeros((x_init.shape[0], len(horizon)))
x = x_init
for i, t in enumerate(horizon):
	p, v = x[:NN*d], x[:-NN*d]

	u = form_maneuv_clv_func(p, v, k_p, k_v, Adj)

	(T, yout, xout) = control.forced_response(sys, X0=x, U=u, return_x=True)

	x_out[i] = xout
	x = xout


# Generate Figure
plt.figure(1)
for x in x_out:
	plt.plot(horizon, x)