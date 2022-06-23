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
def g(p,i,j):
	return (p[j*d:j*d+d] - p[i*d:i*d+d]) / (np.linalg.norm(p[j*d:j*d+d] - p[i*d:i*d+d]) + 10e-15 )

#print(np.linalg.norm(p[7] - p[1]))
#print(p[7] - p[1])
#print(g(0,3))
#exit()

# orthogonal projection matrix P_{g_{ij}}
def P(g_ij):
	return np.identity(d) - g_ij@(g_ij.T)

# formation: square ex. in fig 2 -> agent 1 bottom-left, order counter-clockwise
L = 1
D = np.sqrt(2*L)

#TODO: Set bearing angles instead of positions, leaders must keep position

PP = 	np.array([L,0, L, L, 0, L, 0, 0]).T 
GG = np.zeros((NN, NN, d), dtype=np.float32)
Pg_star = np.zeros((d*NN, d*NN))

#TODO: define pg_star based on the couplets of neighbours being currently checked [i,j,:]
for ii in range(NN):
	for jj in range(NN):
		g_star = g(PP, ii, jj)
		GG[ii, jj, :] = g_star
		Pg_star[ii*d:ii*d+d, jj*d:jj*d+d] = P(g_star)

# TODO: when writing report, use this to demonstrate antisymmetry (it is)
""" is_GG_antisym = GG+np.transpose(GG, axes= (1, 0, 2))
print(is_GG_antisym)
print(GG)
exit() """

p_ER = 0.9

I_NN = np.identity(NN, dtype=int)
I_nx = np.identity(d, dtype=int)
I_NN_nx = np.identity(d*NN, dtype=int)
O_NN = np.ones((NN,1), dtype=int)
	
# ER Network generation
while 1:
	Adj = np.random.binomial(1, p_ER, (NN, NN))
	Adj = np.logical_or(Adj, Adj.T)
	Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)

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
		B[ii*d:ii*d+d, jj*d:jj*d+d] = -Pg_star[ii*d:ii*d+d, jj*d:jj*d+d]
		sumPg_ik_star += Pg_star[ii*d:ii*d+d, jj*d:jj*d+d]
	B[ii*d:ii*d+d, ii*d:ii*d+d] = sumPg_ik_star

print("################## B matrix #####################")
print(np.array_str(B, precision=3, suppress_small=True))

# Partitioning B
n_l = n_leaders
n_f = NN-n_leaders

B_ll = B[d*0:d*n_l, d*0:d*n_l]	# shape (d*n_l, d*n_l)
B_lf = B[d*0:d*n_l, d*n_f:d*NN]	# shape (d*n_l, d*n_f)
B_fl = B[d*n_f:d*NN,d*0:d*n_l]	# shape (d*n_f, d*n_l)
B_ff = B[d*n_f:d*NN,d*n_f:d*NN] # shape (d*n_f, d*n_f) -> if nonsingular then target formation is unique


#we will use these for calculating deltas of position and velocities
pf_star = -np.linalg.inv(B_ff)@B_fl@p[0:n_leaders*d]
#print(pf_star)
vf_star = -np.linalg.inv(B_ff)@B_fl@v[0:n_leaders*d]
#print(vf_star)


#TODO: when wrinting report use this to demonstrate determinant of B_ff != 0 => B unique
#print(np.linalg.det(B_ff))

BB_ext_up = np.concatenate((B_ll, B_lf), axis = 1)
BB_ext_low  = np.concatenate((B_fl, B_ff), axis = 1)
BB = np.concatenate((np.zeros_like(BB_ext_up), BB_ext_low), axis = 0)


# system dynamics: Formation Maneuvering with Constant Leader Velocity
def form_maneuv_clv_func(p, v, k_p, k_v, Adj):
	u = np.zeros(np.shape(v))
	for ii in range(n_f,NN):
		N_ii = np.nonzero(Adj[ii])[0] # In-Neighbors of node i
		for jj in N_ii:
			pp = p[ii*d:ii*d+d] - p[jj*d:jj*d+d]
			vv = v[ii*d:ii*d+d] - v[jj*d:jj*d+d]
			pg_star = Pg_star[ii*d:ii*d+d, jj*d:jj*d+d]
			u[ii*d:ii*d+d] -=  pg_star @ (k_p*pp + k_v*vv)
	return u


L_f = L_IN[(NN-n_leaders):, (NN-n_leaders):]
L_fl = L_IN[(NN-n_leaders):, 0:n_leaders]
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
LL_ext_low = np.concatenate((LL_kron, np.zeros(LL_kron.shape)), axis = 1)
LL_ext = np.concatenate((LL_ext_up, LL_ext_low), axis = 0)

BB = np.concatenate((np.zeros((NN*d,NN*d)), BB), axis = 0)

A = -LL_ext
B = BB
C = np.identity(np.size(LL_ext,axis = 0)) # to comply with StateSpace syntax

################
dt = 0.01
Tmax = 10.0
horizon = np.arange(0.0, Tmax, dt)

sys = control.StateSpace(A,B,C,0) # dx = -L x + B u

k_p = 0.5
k_v = 0.5

""" x_out = np.zeros((x_init.shape[0], len(horizon)))
x = x_init
for i, t in enumerate(horizon):
	p, v = x[:NN*d], x[:-NN*d]

	u = form_maneuv_clv_func(p, v, k_p, k_v, Adj)
	print(u)
	print(len(u))

	(T, yout, xout) = control.forced_response(sys, X0=x, U=u, T=None, return_x=True)

	x_out[i] = xout
	x = xout """
#########################
#try verbose approach to state evolution, creates the state matrix starting from 
#the positon and velocity evolution matrices

""" delta_p = p[(NN-n_leaders)*d:] - pf_star
delta_v = v[(NN-n_leaders)*d:] - vf_star
delta_vect = np.concatenate((delta_p, delta_v), axis=0)
print("##################### delta vector ####################")
print(delta_vect)

state_p = np.concatenate((np.zeros_like(B_ff), np.identity((NN-n_leaders)*2)), axis = 1)
state_v = np.concatenate((-k_p*B_ff, -k_v*B_ff), axis = 1)
state_matrix = np.concatenate((state_p, state_v), axis = 0)
print("##################### state_matrix ####################")
print(state_matrix)

input_p = np.zeros(((NN-n_leaders)*d, (NN-n_leaders)*d))
input_v = -np.linalg.inv(B_ff)@B_fl
input_matrix = np.concatenate((input_p, input_v), axis=0)
print("###################### input matrix #########################")
print(input_matrix)

u = -k_p*B_ff@delta_p -k_v*B_ff@delta_v
#u = np.concatenate((np.zeros(((NN-n_leaders)*d, 1)), u), axis = 0)
print(u)

delta_evo = state_matrix@delta_vect + input_matrix@u
print("###################### delta at time t+1 #########################")
print(delta_evo) """

#start iterations

x_out = np.zeros((x_init.shape[0], len(horizon)))
x = x_init
for i, t in enumerate(horizon):

	p, v = x[:NN*d], x[:-NN*d]

	delta_p = p[(NN-n_leaders)*d:] - pf_star
	delta_v = v[(NN-n_leaders)*d:] - vf_star
	delta_vect = np.concatenate((delta_p, delta_v), axis=0)
	#print("##################### delta vector ####################")
	#print(delta_vect)

	state_p = np.concatenate(
		(np.zeros_like(B_ff), np.identity((NN-n_leaders)*2)), axis=1)
	state_v = np.concatenate((-k_p*B_ff, -k_v*B_ff), axis=1)
	state_matrix = np.concatenate((state_p, state_v), axis=0)
	#print("##################### state_matrix ####################")
	#print(state_matrix)

	input_p = np.zeros(((NN-n_leaders)*d, (NN-n_leaders)*d))
	input_v = -np.linalg.inv(B_ff)@B_fl
	input_matrix = np.concatenate((input_p, input_v), axis=0)
	#print("###################### input matrix #########################")
	#print(input_matrix)

	u = -k_p*B_ff@delta_p - k_v*B_ff@delta_v
	print("############ control signal ###################")
	print(u)

	delta_evo = state_matrix@delta_vect #+ input_matrix@u
	print(f"###################### delta at time t = {t} #########################")
	print(delta_evo)

	zero_vect = np.zeros((n_leaders*d,1))
	dx = np.concatenate((zero_vect, delta_evo[:(NN-n_leaders)*d], \
						zero_vect, delta_evo[:(NN-n_leaders)*d]), axis=0)

	x_out = x - dx
	print("####################### x out ###################")
	print(x_out)
	x = x_out


exit()

#########################
x_out = np.zeros((x_init.shape[0], len(horizon)))
x = x_init
for i, t in enumerate(horizon):
	p, v = x[:NN*d], x[:-NN*d]

	u = form_maneuv_clv_func(p, v, k_p, k_v, Adj)

	# dx = -L x + B u
	dx = A@x + B@u

	print("P")
	print(p)
	print("B x U")
	print(B@u)
	print("A x X")
	print(A@x)

	print(f"---------\n{dx}\n---------")

	x_out = x + dx
	
	x = x_out

exit()
# Generate Figure
plt.figure(1)
for x in x_out:
	plt.plot(horizon, x)