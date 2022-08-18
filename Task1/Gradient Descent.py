import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

###############################################################################
# Useful constants
MAXITERS = 20 + 1
NN = 5  # Agents
T = 5   # Layers
chosen_class = 4    # Class to predict
n_samples = NN*40   # Number of training samples

###############################################################################
# Generate Network Binomial Graph
p_ER = 0.9
I_NN = np.eye(NN) # np.identity
while 1:
	Adj = np.random.binomial(1,p_ER, (NN,NN))
	Adj = np.logical_or(Adj,Adj.T)
	Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)

	test = np.linalg.matrix_power(I_NN+Adj, NN)
	if np.all(test>0):
		break

###############################################################################
# Compute weighted adjacency matrix
WW = np.zeros((NN,NN))

for ii in range(NN):
  N_ii = np.nonzero(Adj[ii])[0] # In-Neighbors of node i
  deg_ii = len(N_ii)
  
  for jj in N_ii:
    N_jj = np.nonzero(Adj[jj])[0] # In-Neighbors of node j
    # deg_jj = len(N_jj)
    deg_jj = N_jj.shape[0]

    WW[ii,jj] = 1/(1+max( [deg_ii,deg_jj] ))
    # WW[ii,jj] = 1/(1+np.max(np.stack((deg_ii,deg_jj)) ))

WW += I_NN - np.diag(np.sum(WW,axis=0))

# Compute mixing matrix
# threshold = 1e-10
# WW = 1.5*I_NN + 0.5*Adj

# ONES = np.ones((NN,NN))
# ZEROS = np.zeros((NN,NN))
# WW = np.maximum(WW,0*ONES)
# while any(abs(np.sum(WW,axis=1)-1) > threshold):
# 	WW = WW/(WW@ONES) # row
# 	# WW = WW/(ONES@WW) # col
# 	WW = np.maximum(WW,0*ONES)

print('Check Stochasticity:\n row: {} \n column {}\n'.format(
	np.sum(WW,axis=1),
	np.sum(WW,axis=0)
))
###############################################################################

FF = np.zeros((MAXITERS))

# Activation Function
def sigmoid_fn(xi):
	return 1/(1+np.exp(-xi))

# Derivative of Activation Function
def sigmoid_fn_derivative(xi):
	return sigmoid_fn(xi)*(1-sigmoid_fn(xi))

# Inference: x_tp = f(xt,ut)
def inference_dynamics(xt,ut):
	"""
	input: 
				xt current state
				ut current input
	output: 
				xtp next state
	"""
	xtp = np.zeros(d)
	for ell in range(d):
		temp = xt@ut[ell,1:] + ut[ell,0] # including the bias
		xtp[ell] = sigmoid_fn( temp ) # x' * u_ell
	
	return xtp

# Forward Propagation
def forward_pass(uu,x0):
	"""
	input: 
				uu input trajectory: u[0],u[1],..., u[T-1]
				x0 initial condition
	output: 
				xx state trajectory: x[1],x[2],..., x[T]
	"""
	xx = np.zeros((T, d))
	xx[0] = x0

	for t in range(T-1):
		xx[t+1] = inference_dynamics(xx[t], uu[t]) # x^+ = f(x,u)

	return xx

# Adjoint dynamics: 
#   state:    lambda_t = A.T lambda_tp
#   output:   deltau_t = B.T lambda_tp
def adjoint_dynamics(ltp, xt, ut):
	"""
		input: 
				llambda_tp current costate
				xt current state
				ut current input
		output: 
				llambda_t next costate
				delta_ut loss gradient wrt u_t
	"""
	df_dx = np.zeros((d,d))
	# df_du = np.zeros((d,(d+1)*d))
	
	Delta_ut = np.zeros((d,d+1))

	for j in range(d):
		dsigma_j = sigmoid_fn_derivative(xt@ut[j,1:] + ut[j,0]) 

		df_dx[:,j] = ut[j,1:]*dsigma_j
		# df_du[j, XX] = dsigma_j*np.hstack([1,xt])
		
		# B'@ltp
		Delta_ut[j,0] = ltp[j]*dsigma_j
		Delta_ut[j,1:] = xt*ltp[j]*dsigma_j
	
	lt = df_dx@ltp # A'@ltp
	# Delta_ut = df_du@ltp

	return lt, Delta_ut

# Backward Propagation
def backward_pass(xx,uu,llambdaT):
	"""
		input: 
				xx state trajectory: x[1],x[2],..., x[T]
				uu input trajectory: u[0],u[1],..., u[T-1]
				llambdaT terminal condition
		output: 
				llambda costate trajectory
				delta_u costate output, i.e., the loss gradient
	"""
	llambda = np.zeros((T,d))
	llambda[-1] = llambdaT

	Delta_u = np.zeros((T-1,d,d+1))

	for t in reversed(range(T-1)): # T-1,T-2...,1,0
		llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t+1],xx[t],uu[t])

	return Delta_u

# Mean-Squared Error - Loss Function
def MSE(y_pred, y_true):
	mse = (y_pred - y_true)**2
	mse_d = 2*(y_pred - y_true)
	return mse, mse_d

# Bunary Cross-Entropy - Loss Function
def BCE(y_pred, y_true):
	bce = - (y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
	bce_d = - (y_true - y_pred) / (y_pred*(1 - y_pred) + 1e-15)
	return bce, bce_d

###############################################################################
# Dataset preparation

(train_D, train_y), (test_D, test_y) = mnist.load_data()
train_D, test_D = train_D/255.0, test_D/255.0
#print(train_D.shape, test_D.shape)

train_D = train_D.reshape((train_D.shape[0], 28 * 28))
test_D = test_D.reshape((test_D.shape[0], 28 * 28))

train_y = [1 if y == chosen_class else 0 for y in train_y]
test_y =  [1 if y == chosen_class else 0 for y in test_y]

# Shuffle
idx = np.arange(train_D.shape[0])
np.random.shuffle(idx)
train_D = np.array([train_D[i] for i in idx])
train_y = np.array([train_y[i] for i in idx])

# Without sampling: the probability distribution of chosen_class is 1/10
# n = n_samples//NN # Number of samples per node
# data_point = np.array([train_D[n*i: n*i+n] for i in range(NN)])
# label_point = np.array([train_y[n*i: n*i+n] for i in range(NN)])

# With pos/neg sampling: the proability distribution of chosen_class is 1/2
pos_idx = np.where(train_y == 1)[0][:n_samples//2]
neg_idx = np.where(train_y == 0)[0][:n_samples-n_samples//2]

n = n_samples//NN # Number of samples per node
indices = np.concatenate((pos_idx, neg_idx))
np.random.shuffle(indices)
indices = indices.reshape((NN, n))

data_point = train_D[indices]
label_point = train_y[indices]

print(f"Training label points:\n{label_point}\n")

# Training

d = 28*28           # Number of neurons in each layer. Same numbers for all the layers
stepsize = 1e-4     # Learning rate
J = np.zeros((MAXITERS))  # Cost

UU = np.random.randn(NN, T-1, d, d+1)	# U_t :Initial Weights / Initial Input Trajectory
UUp = np.zeros_like(UU)					# U_{t+1}
VV = np.zeros_like(UU)

Delta_u = np.zeros_like(UU)

print(f"Training...")
for tt in range(MAXITERS):  # For each iteration

	for ii in range(NN):  # For each node
		totalCost = 0  # Sum up the cost of each image
		Delta_u[ii] = 0
		for kk in range(n):  # For each image of the node

			image = data_point[ii][kk]
			label = label_point[ii][kk]

			# Forward pass
			XX = forward_pass(UU[ii], image)  # f_i(x_i,t)
			# Cost function
			cost, cost_d = BCE(XX[-1,-1], label)
			totalCost += cost
			llambdaT = cost_d

			# Backward propagation            # \nabla f_i(x_{i,t})
			Delta_u[ii] += backward_pass(XX, UU[ii], llambdaT) # Sum weigth errors on the full batch of images

		# Store the Loss Value across Iterations (the sum of costs of all nodes)
		J[tt] += totalCost
	
	for ii in range(NN):
		Nii = np.nonzero(Adj[ii])[0] # Self loop is not present

		# Weights update
		VV[ii] = WW[ii,ii]*UU[ii]
		for jj in Nii:
			VV[ii] += WW[ii,jj]*UU[jj]
		#UUp[ii] = VV[ii] - stepsize*Delta_u				# constant step-size
		UUp[ii] = VV[ii] - stepsize*Delta_u[ii]/(tt+1)*10	# diminishing step-size

	# Update the current step
	UU = UUp

	if (tt % 1) == 0:
		print(f"Iteration {tt:3d} - loss: {J[tt]:4.3f}", end="\n")

# Evolution of cost function
fig = plt.figure()
plt.semilogy(np.arange(MAXITERS), J, linestyle='-', linewidth=2)
plt.xlabel(r"iterations $t$")
plt.ylabel(r"cost")
plt.title(r"Evolution of the cost error: $\min \sum_{i=1}^N \sum_{k=1}^\mathcal{I} J(\phi(u;x_i^k);y_i^k)$")
plt.grid()
plt.show()
#fig.savefig('./imgs/Gradient Tracking.png')

# Evaluation on test set
y_pred = []

idx = np.argsort(np.random.random(test_D.shape[0]))
test_images = [test_D[i] for i in idx][:n_samples]
test_labels = [test_y[i] for i in idx][:n_samples]
for image, label in zip(test_images, test_labels):
	# Forward pass
	XX = forward_pass(UU[ii], image)  # f_i(x_i,t)
	y_pred.append(1 if XX[-1, -1] > 0.5 else 0)

print(y_pred)
print(test_labels)

print()

print(f'accuracy: {accuracy_score(test_labels, y_pred):.4f}')
weights = [1 - test_labels.count(1) / len(test_labels) if i == 1 else 1 -
		   test_labels.count(0) / len(test_labels) for i in test_labels]
print(
	f'accuracy with weights: {accuracy_score(test_labels, y_pred, sample_weight=weights):.4f}')