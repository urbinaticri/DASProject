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
	return (p[j,j+d] - p[i:i+d]) / np.linalg.norm(p[j,j+d] - p[i:i+d])

# orthogonal projection matrix P_{g_{ij}}
def P(g_ij):
	return np.identity(d) - g_ij@(g_ij.t)

# formation: square ex. in fig 2 -> agent 1 bottom-left, order counter-clockwise
L = 1
D =sqrt(1)
g_star = [[	[0,0],		[0,L],		[D,D],		[L,0]],
	 	  [	[0,L],		[0,0],		[1,L],		[D,D]],
	 	  [	[D.D],		[L,0],		[0,0], 		[0,L]],
	 	  [	[L,0]		[D,D],		[0,L]		[0,0]]]
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


# system dynamics: Formation Maneuvering with Constant Leader Velocity
def form_maneuv_clv_func(p, v, k_p, k_v, Adj):
	u = np.zeros(np.size(v))
	for ii in range(NN):
		N_ii = np.where(Adj[:,ii]>0)[0]
		index_ii =  ii*d + np.arange(d)
		for jj in N_ii:
			index_jj = jj*d + np.arange(d)
			pp = p[index_ii] - p[index_jj]
			vv = v[index_ii] - v[index_jj]
			u -= P(g_star[index_ii, index_jj]) * (k_p*pp + k_v*vv)
	return u


##############################################
# QUI SOTTO CODICE DEL PROF ANCORA DA CAPIRE #
##############################################


L_f = L_IN[0:NN-n_leaders, 0:NN-n_leaders]
L_fl = L_IN[0:NN-n_leaders, NN-n_leaders:]

################
# leaders dynamics
LL = np.concatenate((L_f, L_fl), axis = 1)
LL = np.concatenate((LL, np.zeros((n_leaders,NN))), axis = 0)

# replicate for each dimension
LL_kron = np.kron(LL,I_nx)

x_init = np.vstack((
	np.ones((d*n_leaders,1)),
	np.zeros((d*(NN-n_leaders),1))
))
x_init += 5*np.random.rand(d*NN,1)

BB_kron = np.zeros((NN*d, n_leaders*d))
BB_kron[(NN-n_leaders)*d:,:] = np.identity(d*n_leaders, dtype=int)

A = -LL_kron
B = BB_kron
C = np.identity(np.size(LL_kron,axis = 0)) # to comply with StateSpace syntax

################
## followers integral Action

k_i = 4
K_I = - k_i*I_NN_nx

LL_ext_up = np.concatenate((LL_kron, K_I), axis = 1)
LL_ext_low = np.concatenate((LL_kron, np.zeros(LL_kron.shape)), axis = 1)
LL_ext = np.concatenate((LL_ext_up, LL_ext_low), axis = 0)

# include integral state
x_init = np.concatenate((x_init,np.zeros((d*NN,1))))
BB_kron = np.concatenate((BB_kron, np.zeros((NN*d,n_leaders*d))), axis = 0)

A = -LL_ext
B = BB_kron
C = np.identity(np.size(LL_ext,axis = 0)) # to comply with StateSpace syntax

################

sys = control.StateSpace(A,B,C,0) # dx = -L x + B u

dt = 0.01
Tmax = 10.0
horizon = np.arange(0.0, Tmax, dt)

# Leaders input
velocity = 0
u = velocity*np.ones((d*n_leaders, len(horizon)))

# da sostituire con formation function
#(T, yout, xout) = control.forced_response(sys, X0=x_init, U=u, T=horizon, return_x=True)

# numerical integration
fc_dynamics = lambda t,x: form_func(t, x, distances, Adj, d)

# Solve an initial value problem for a system of ODEs
res = solve_ivp(fc_dynamics,		# function to integrate
				t_span = [0, Tmax],	# time limits # t_span = [0, Tmax]
				y0 = x_init			# initial condition
				)
xout = res.y
T = res.t

# Generate Figure
plt.figure(1)
for x in xout:
	plt.plot(T, x)

# Plot mean values
# plt.plot(T, np.repeat(x0_mean, len(T), axis = 0),  '--', linewidth=3)

plt.title("Evolution of the local estimates")
plt.xlabel("$t$")
plt.ylabel("$x_i^t$")

if 1: # animation (0 to avoid animation)
	if d == 2: 
		plt.figure(2)
		animation(xout,NN,d,n_leaders, horizon = T, dt=10)

plt.show()
