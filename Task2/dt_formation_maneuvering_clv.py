#
# Discrete Time Formation Maneuvering with Constant Leader Velocity
#
import numpy as np
import time
import matplotlib.pyplot as plt
from animations import formation as animation

ANIMATION = True
np.random.seed(5)

filename = "formation_D" # Filename with the formation to obtain
NN = 8 # number of agents
n_leaders = 2 # number of leaders
d = 2 # space dimension

def read_file(filename, n_agents):
    PP = []
    Adj = []
    cntr = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            if cntr < n_agents: #PP
                PP.extend([float(p) for p in line.split()])
            else: #Adj
                Adj.append([float(p) for p in line.split()])
            cntr += 1
    return np.array(PP), np.asarray(Adj)

# Load positions and adjacency matrix of desired formation
L = 1.0 # Scale
PP, Adj = read_file(filename, NN)
PP = L*PP

# initial positions
p = np.vstack((
	np.array(PP[:n_leaders*d]).reshape(d*n_leaders, 1),
	np.zeros((d*(NN-n_leaders),1)) + 5*np.random.rand(d*(NN-n_leaders),1)
))

# initial velocities
constant_v = 0 #Leaders constant velocity
v = np.vstack((
	np.zeros((d*n_leaders,1)) + constant_v,
	np.zeros((d*(NN-n_leaders),1))
))

# state vector initialization for all agents
x_init = np.vstack((
	p,
	v
))

# bearing unit vector g_{ij} function
def g(pp,ii,jj):
	index_ii = ii*d + np.arange(d)
	index_jj = jj*d + np.arange(d)
	return (pp[index_jj] - pp[index_ii]) / (np.linalg.norm(pp[index_jj] - pp[index_ii]) + 1e-15)

# orthogonal projection matrix P_{g_{ij}} function
def P(g_ij):
	g_ij = g_ij.reshape((-1, 1)) # here reshape because from row array i.e. [1,0] we want col array i.e. [[1], [0]] 
	return np.identity(d) - g_ij@(g_ij.T)

GG = np.zeros((NN, NN, d), dtype=np.float32) # matrix containing bearing vector of desired formation
Pg_star = np.zeros((NN, NN, d, d), dtype=np.float32) # projection matrix of desired formation

for ii in range(NN):
	for jj in range(NN):
		g_star = g(PP, ii, jj)
		GG[ii, jj, :] = g_star
		Pg_star[ii, jj, :] = P(g_star)

# When writing report, use this to demonstrate antisymmetry: g_{ji} = - g_{ij}
is_GG_antisym = not np.any(GG+np.transpose(GG, axes= (1, 0, 2)))
print(is_GG_antisym)

# Bearing Laplacian Matrix
B = np.zeros((d*NN, d*NN))
for ii in range(NN):
	N_ii = np.nonzero(Adj[ii])[0] # In-Neighbors of node i
	index_ii =  ii*d + np.arange(d)
	sumPg_ik_star = 0
	for jj in N_ii:
		index_jj = jj*d + np.arange(d)
		B[ii*d:ii*d+d, jj*d:jj*d+d] = -Pg_star[ii, jj]
		sumPg_ik_star += Pg_star[ii, jj]
	B[ii*d:ii*d+d, jj*d:jj*d+d] = sumPg_ik_star

n_l = n_leaders
n_f = NN-n_leaders

B_ll = B[d*0:d*n_l, d*0:d*n_l]	# shape (d*n_l, d*n_l)
B_lf = B[d*0:d*n_l, d*n_f:d*NN]	# shape (d*n_l, d*n_f)
B_fl = B[d*n_f:d*NN,d*0:d*n_l]	# shape (d*n_f, d*n_l)
B_ff = B[d*n_f:d*NN,d*n_f:d*NN] # shape (d*n_f, d*n_f) -> if nonsingular then target formation is unique

#When wrinting report use this to demonstrate determinant of B_ff != 0 => Target formation is unique
print(np.linalg.det(B_ff) != 0)

# system dynamics: Formation Maneuvering with Constant Leader Velocity
def form_maneuv_clv_func(p, v, k_p, k_v, Adj):
	u = np.zeros(np.shape(v))
	for ii in range(n_l,NN):
		N_ii = np.nonzero(Adj[ii])[0] # In-Neighbors of node i
		index_ii =  ii*d + np.arange(d)
		for jj in N_ii:
			index_jj = jj*d + np.arange(d)
			pp = p[index_ii] - p[index_jj]
			vv = v[index_ii] - v[index_jj]
			pg_star = Pg_star[ii, jj]
			u[index_ii] -=  pg_star @ (k_p*pp + k_v*vv)
	return u

#create zeros and one matrices used to build A and B matrices
A_ll = np.zeros((NN*d, NN*d))
A_lf = np.eye(NN*d)
A_fl = np.zeros((NN*d, NN*d))
A_ff = np.zeros((NN*d, NN*d))

A_top = np.concatenate((A_ll, A_lf), axis = 1)
A_low = np.concatenate((A_fl, A_ff), axis = 1)
A = np.concatenate((A_top, A_low), axis = 0)
# print(f"A matrix, shape:{A.shape}")
# print(np.array_str(A, precision=2, suppress_small=True))

B_top = np.zeros((NN*d, NN*d))
B_low = np.eye(NN*d)
B = np.concatenate((B_top, B_low), axis = 0)
# print(f"B matrix, shape:{B.shape}")
# print(np.array_str(B, precision=2, suppress_small=True))

# Discrete-Time system control
dt = 0.1
Tmax = 20.0
horizon = np.arange(0.0, Tmax, dt)
xout = []
PP_curr = PP
dist_err = []
T = []

k_p = 4
k_v = 4

x = x_init
for i, t in enumerate(horizon):
	p, v = x[:NN*d], x[NN*d:]

	# Append position and time for final plot
	xout.append(np.copy(p.reshape((-1,)))) 
	T.append(t)
	
	dist_err.append(p.reshape((-1,)) - PP_curr)
	PP_curr += constant_v*dt

	u = form_maneuv_clv_func(p, v, k_p, k_v, Adj)
	# print(f"u matrix, shape:{u.shape}")
	# print(np.array_str(u, precision=2, suppress_small=True))

	dx = A@x + B@u
	# print(f"dx matrix, shape:{dx.shape}")
	# print(np.array_str(dx, precision=2, suppress_small=True))

	x += dx*dt
	# print(f"x matrix, shape:{x.shape}")
	# print(np.array_str(x, precision=2, suppress_small=True))


print(f"x matrix, shape:{x.shape}")
print(np.array_str(x, precision=2, suppress_small=True))

xout = np.array(xout).T
T = np.array(T)
dist_err = np.array(dist_err).T

plt.figure(1)
for ii in range(NN*d):
	plt.plot(T, dist_err[ii,:])
plt.xlabel('$t$', fontsize=16)
plt.title("Distance error w.r.t. desired coordinates")

if ANIMATION: # animation (0 to avoid animation)
	plt.figure(2)
	animation(xout,T,Adj,NN,d)
time.sleep(10000)