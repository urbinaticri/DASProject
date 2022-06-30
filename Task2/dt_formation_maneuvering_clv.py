import numpy as np
import matplotlib.pyplot as plt
from animations import formation as animation

ANIMATION = True
np.random.seed(5)

NN = 4 # number of agents
n_leaders = 2 # number of leaders
d = 2 # dimension of positions and velocities

# formation: square ex. in fig 2 -> agent 1 bottom-left, order counter-clockwise
<<<<<<< HEAD
L = 1.0
PP = np.array([L, 0, L, L, 0, L, 0, 0])

# initial positions
p = np.vstack((
	np.array([L, 0, L, L]).reshape(d*n_leaders, 1),
	np.zeros((d*(NN-n_leaders),1)) + 5*np.random.rand(d*(NN-n_leaders),1)
))
=======
L = 1
D = np.sqrt(2*L)

# positions
p = np.array([[L, 0, L, L, 5, -4, 2, 0.5]]).T
""" p = np.vstack((
	np.array([[L,0, L, L]]).T,
	np.zeros((d*(NN-n_leaders),1) + 5*np.random.rand(d*(NN-n_leaders),1))
)) """
>>>>>>> f01defae7ff0859839fa4252228642af304dcfd1

# initial velocities
constant_v = 0.5 #Leaders constant velocity
v = np.vstack((
	np.zeros((d*n_leaders,1)) + constant_v,
	np.zeros((d*(NN-n_leaders),1))
))

x_init = np.vstack((
	p,
	v
))

# bearing unit vector g_{ij}
def g(pp,ii,jj):
	index_ii = ii*d + np.arange(d)
	index_jj = jj*d + np.arange(d)
	return (pp[index_jj] - pp[index_ii]) / (np.linalg.norm(pp[index_jj] - pp[index_ii]) + 1e-15)

# orthogonal projection matrix P_{g_{ij}}
def P(g_ij):
	g_ij = g_ij.reshape((-1, 1)) #here reshape because from row array i.e. [1,0] we want col array i.e. [[1], [0]] 
	return np.identity(d) - g_ij@(g_ij.T)


#TODO: Set bearing angles instead of positions, leaders must keep position

PP = np.array([L,0, L, L, 0, L, 0, 0]).T 
GG = np.zeros((NN, NN, d), dtype=np.float32)
Pg_star = np.zeros((NN, NN, d, d), dtype=np.float32)

for ii in range(NN):
	for jj in range(NN):
		g_star = g(PP, ii, jj)
		GG[ii, jj, :] = g_star
		Pg_star[ii, jj, :] = P(g_star)

<<<<<<< HEAD
# TODO: when writing report, use this to demonstrate antisymmetry
=======
""" print(np.array_str(Pg_star, precision=5, suppress_small=True))
print(np.array_str(GG, precision=5, suppress_small=True))
exit() """

# TODO: when writing report, use this to demonstrate antisymmetry (it is)
>>>>>>> f01defae7ff0859839fa4252228642af304dcfd1
""" is_GG_antisym = GG+np.transpose(GG, axes= (1, 0, 2))
print(is_GG_antisym)
print(GG)
exit() """

# ER Network generation
p_ER = 0.9

I_NN = np.identity(NN, dtype=int)
I_nx = np.identity(d, dtype=int)
I_NN_nx = np.identity(d*NN, dtype=int)
O_NN = np.ones((NN,1), dtype=int)

while 1:
	Adj = np.random.binomial(1, p_ER, (NN, NN))
	Adj = np.logical_or(Adj, Adj.T)
	Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)

	# test connectivity
	test = np.linalg.matrix_power((I_NN+Adj),NN)
	
	if np.all(test>0):
		print("the graph is connected\n")
		break

# Adj = np.asarray([
# 	[0,     1,      1,		1],
# 	[1,     0,      1,    	0],
# 	[1,     1,      0,    	1],     
# 	[1,     0,      1,    	0]
# ])
DEGREE = np.sum(Adj, axis=0)
D_IN = np.diag(DEGREE)
# Laplacian Matrix
L_IN = D_IN - Adj.T

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

#TODO: when wrinting report use this to demonstrate determinant of B_ff != 0 => B unique
#print(np.linalg.det(B_ff))


# system dynamics: Formation Maneuvering with Constant Leader Velocity
def form_maneuv_clv_func(p, v, k_p, k_v, Adj):
	u = np.zeros(np.shape(v))
	for ii in range(n_f,NN):
		N_ii = np.nonzero(Adj[ii])[0] # In-Neighbors of node i
		index_ii =  ii*d + np.arange(d)
		for jj in N_ii:
			index_jj = jj*d + np.arange(d)
			pp = p[index_ii] - p[index_jj]
			vv = v[index_ii] - v[index_jj]
			pg_star = Pg_star[ii, jj]
			u[index_ii] -=  pg_star @ (k_p*pp + k_v*vv)
	return u


# # Leader-Follower Laplacian Dynamics
# L_f = L_IN[(NN-n_leaders):, (NN-n_leaders):]
# L_fl = L_IN[(NN-n_leaders):, 0:n_leaders]
# LL = np.concatenate((L_f, L_fl), axis = 1)
# LL = np.concatenate((np.zeros((n_leaders,NN)), LL), axis = 0)

# # replicate for each dimension
# LL_kron = np.kron(LL,I_nx)

# # Followers ext with integral Action

# k_i = 0.4
# K_I = -k_i*I_NN_nx

# LL_ext_up = np.concatenate((LL_kron, K_I), axis = 1)
# LL_ext_low = np.concatenate((LL_kron, np.zeros(LL_kron.shape)), axis = 1)
# LL_ext = np.concatenate((LL_ext_up, LL_ext_low), axis = 0)


#create zeros and one matrices used to build A and B matrices
A_ll = np.zeros((NN*d, NN*d))
A_lf = np.eye(NN*d)
A_fl = np.zeros((NN*d, NN*d))
A_ff = np.zeros((NN*d, NN*d))

<<<<<<< HEAD
A_top = np.concatenate((A_ll, A_lf), axis = 1)
A_low = np.concatenate((A_fl, A_ll), axis = 1)
A = np.concatenate((A_top, A_low), axis = 0)
# print(f"A matrix, shape:{A.shape}")
# print(np.array_str(A, precision=2, suppress_small=True))

B_top = np.zeros((NN*d, NN*d))
B_low = np.eye(NN*d)
B = np.concatenate((B_top, B_low), axis = 0)
# print(f"B matrix, shape:{B.shape}")
# print(np.array_str(B, precision=2, suppress_small=True))
=======
#create zeros and one matrices used to build A and B matrices
zero_mff = np.zeros((NN*d, NN*d))
zero_mfl = np.zeros((NN*d, NN*d))
zero_mlf = np.zeros((NN*d, NN*d))
zero_mll = np.zeros((NN*d, NN*d))
one_mff = np.eye(NN*d)


A_top = np.concatenate((zero_mll, zero_mff), axis = 1)
A_bottom = np.concatenate((zero_mfl, one_mff), axis = 1)
A= np.concatenate((A_top, A_bottom), axis = 0)
print(A)
B = np.concatenate((zero_mll, one_mff), axis = 0)
print(B)
C = np.identity(np.size(LL_ext,axis = 0)) # to comply with StateSpace syntax
>>>>>>> f01defae7ff0859839fa4252228642af304dcfd1

################
dt = 0.1
Tmax = 20.0
horizon = np.arange(0.0, Tmax, dt)
xout = []
PP_curr = PP
dist_err = []
T = []

k_p = 1
k_v = 1

<<<<<<< HEAD
=======
	u = form_maneuv_clv_func(p, v, k_p, k_v, Adj)
	print(u)
	print(len(u))

	(T, yout, xout) = control.forced_response(sys, X0=x, U=u, T=None, return_x=True)

	x_out[i] = xout
	x = xout """
#########################
#try verbose approach to state evolution, creates the state matrix starting from 
#the positon and velocity evolution matrices

"""
#start iterations

x_out = np.zeros((x_init.shape[0], len(horizon)))
>>>>>>> f01defae7ff0859839fa4252228642af304dcfd1
x = x_init
for i, t in enumerate(horizon):
	p, v = x[:NN*d], x[NN*d:]

<<<<<<< HEAD
	# Append position and time for final plot
	xout.append(np.copy(p.reshape((-1,)))) 
	T.append(t)
	
	dist_err.append(p.reshape((-1,)) - PP_curr)
	PP_curr += constant_v*dt

	u = form_maneuv_clv_func(p, v, k_p, k_v, Adj)
	# print(f"u matrix, shape:{u.shape}")
	# print(np.array_str(u, precision=2, suppress_small=True))

=======
	p, v = x[:NN*d], x[:-NN*d]

	delta_p = p[(NN-n_leaders)*d:] - pf_star
	delta_v = v[(NN-n_leaders)*d:] - vf_star
	delta_vect = np.concatenate((delta_p, delta_v), axis=0)
	#print("##################### delta vector ####################")
	#print(delta_vect)

	state_p = np.concatenate((np.zeros_like(B_ff), \
							 np.identity((NN-n_leaders)*2)), axis=1)
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
	dx = np.vstack((zero_vect, delta_evo[:(NN-n_leaders)*d], \
						zero_vect, delta_evo[(NN-n_leaders)*d:]))
	x_out[:,i] = x + dx
	print("####################### x out ###################")
	print(x_out)
	x = x_out[:,i]
"""

#########################
x_out = np.zeros((x_init.shape[0], len(horizon)))
x = x_init
for i, t in enumerate(horizon):
	p, v = x[:NN*d], x[:-NN*d]

	u = form_maneuv_clv_func(p, v, k_p, k_v, Adj)
	print(u)
>>>>>>> f01defae7ff0859839fa4252228642af304dcfd1
	dx = A@x + B@u
	# print(f"dx matrix, shape:{dx.shape}")
	# print(np.array_str(dx, precision=2, suppress_small=True))

<<<<<<< HEAD
	x += dx*dt
	# print(f"x matrix, shape:{x.shape}")
	# print(np.array_str(x, precision=2, suppress_small=True))
=======
	""" dv = u
	dp = v

	v += dv
	p += dp

	x = np.concatenate((p,v), axis=0) """

	print("P")
	print(p)
	print("B x U")
	print(B@u)
	print("A x X")
	print(A@x)
>>>>>>> f01defae7ff0859839fa4252228642af304dcfd1


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
plt.show()