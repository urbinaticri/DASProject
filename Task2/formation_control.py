#
# Formation Control Algorithm
# Lorenzo Pichierri, Andrea Testa
# Bologna, 30/03/2022
#
import numpy as np
import matplotlib.pyplot as plt
from animations import formation as animation
from scipy.integrate import solve_ivp
# np.random.seed(10)

ANIMATION = True

#####################################
# Functions
#####################################

# system dynamics
def form_func(t, x, dist, Adj, n_x):
	xdot = np.zeros(np.size(x))
	NN = np.size(Adj,1)
	for ii in range(NN):
		N_ii = np.where(Adj[:,ii]>0)[0]
		index_ii =  ii*n_x + np.arange(n_x)
		for jj in N_ii:
			index_jj = jj*n_x + np.arange(n_x)
			xx_ii = x[index_ii]
			xx_jj = x[index_jj]
			dV_ij = (np.linalg.norm(xx_ii - xx_jj)**2 - dist[ii,jj]**2) * (xx_ii - xx_jj)
			xdot[index_ii] = xdot[index_ii] - dV_ij
	return xdot

# function to evaluate the error (for plots)
def dist_error(NN, distances, horizon, x):
	TT = np.size(horizon,0)
	err = np.zeros((distances.shape[0], distances.shape[1], TT))

	for tt in range(TT):
		for ii in range(NN):
			N_ii = np.where(Adj[:,ii]>0)[0]
			index_ii =  ii*n_x + np.arange(n_x)
			for jj in N_ii:
				index_jj = jj*n_x + np.arange(n_x)
				xx_ii = x[index_ii,tt]
				xx_jj = x[index_jj,tt]
				norm_ij = np.linalg.norm(xx_ii-xx_jj)

				#relative error
				err[ii,jj,tt] = distances[ii,jj] - norm_ij
	return err

#####################################
# Main Code
#####################################

np.random.seed(1) # seed to replicate simulations

NN = 6 # number of agents
n_x = 2 # dimension of x_i 

# Weight matrix to control inter-agent distances
L = 2
D = 2*L
H = np.sqrt(3)*L

# minimally rigid 2*N-3 (only for regular polygons)
# rigid
distances = [[0,     L,      0,    D,     H,    L],
			[L,     0,      L,    0,     D,    0],
			[0,     L,      0,    L,     0,    D],     
			[D,     0,      L,    0,     L,    0],     
			[H,     D,      0,    L,     0,    L],     
			[L,     0,      D,    0,     L,    0]]

distances = np.asarray(distances) #convert list to numpy array

# Adjacency matrix
Adj = distances > 0

# definite initial positions
x_init = np.random.rand(n_x*NN)

dt = 0.1 # integration time
Tmax = 10.0 # simulation time
horizon = np.linspace(0.0, Tmax, int(Tmax/dt))

# numerical integration
fc_dynamics = lambda t,x: form_func(t, x, distances, Adj, n_x)

# Solve an initial value problem for a system of ODEs
res = solve_ivp(fc_dynamics,		# function to integrate
				t_span = [0, Tmax],	# time limits # t_span = [0, Tmax]
				y0 = x_init			# initial condition
				)
xout = res.y
T = res.t

# Evaluate the distance error
err = dist_error(NN, distances, T, xout)
dist_err = np.reshape(err,(NN*NN,np.size(T)))

# generate figure
plt.figure(1)

for ii in range(NN*NN):
	plt.plot(T, dist_err[ii,:])

plt.xlabel('$t$', fontsize=16)
plt.ylabel('$||x_i^t-x_j^t||-d_{ij}, i = 1,...,N$', fontsize=16)

if ANIMATION: # animation (0 to avoid animation)
	plt.figure(2)
	animation(xout,T,Adj,NN,n_x)

plt.show()