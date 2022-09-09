import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

###############################################################################
# Useful constants
MAXITERS = 50 + 1
NN = 5  # Agents
T = 3   # Layers
chosen_class = 4    # Class to predict
n_samples = NN*20   # Number of training samples

###############################################################################
# Generate Network Binomial Graph
p_ER = 0.8
I_NN = np.eye(NN)  # np.identity
while 1:
	Adj = np.random.binomial(1, p_ER, (NN, NN)) # Generates a NNxNN matrix drawing values from a binomial distribution
	Adj = np.logical_or(Adj, Adj.T) # Makes the matrix symmetric
	Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int) # Set 0 on main diagonal

	test = np.linalg.matrix_power(I_NN+Adj, NN) # Strongly connected graph test
	if np.all(test > 0):
		break

# Compute mixing matrix
threshold = 1e-10
WW = 1.5*I_NN + 0.5*Adj

ONES = np.ones((NN,NN))
ZEROS = np.zeros((NN,NN))
WW = np.maximum(WW,0*ONES)
while any(abs(np.sum(WW,axis=1)-1) > threshold):
	WW = WW/(WW@ONES) # Row-stochasticity
	WW = WW/(ONES@WW) # Col-stochasticity
	WW = np.maximum(WW,0*ONES)

print('Check Stochasticity:\n row: {} \n column {}\n'.format(
	np.sum(WW,axis=1),
	np.sum(WW,axis=0)
))
###############################################################################

# Inference: x_tp = f(xt,ut)
def inference_dynamics(xt, ut):
	"""
	input: 
							xt current state    i.e. output of the layer L-1
							ut current input    i.e. weights of the layer L
	output: 
							xtp next state      i.e. i.e. output of the layer L
	"""
	xtp = np.zeros(d)
	for ell in range(d):
		xtp[ell] = xt@ut[ell, 1:] + ut[ell, 0]

	return xtp

# Forward Propagation
def forward_pass(uu, x0):
	"""
	input: 
							uu input trajectory: u[0],u[1],..., u[T-1]  i.e. weights of the neural network
							x0 initial condition                        i.e. input image
	output: 
							xx state trajectory: x[1],x[2],..., x[T]    i.e. output of the neural network
	"""
	xx = np.zeros((T, d))
	xx[0] = x0

	for t in range(T-1):
		xx[t+1] = inference_dynamics(xx[t], uu[t])  #  x^+ = f(x,u)

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
    df_dx = np.zeros((d, d))
    df_du = np.zeros((d, (d+1)))

    Delta_ut = np.zeros((d, d+1))

    for j in range(d):
        df_dx[:, j] = ut[j, 1:]
        df_du[j, :] = np.hstack([1, xt])

        # B'@ltp
        Delta_ut[j, 0] = df_du[j, 0]*ltp[j]
        Delta_ut[j, 1:] = df_du[j, 1:]*ltp[j]

    lt = df_dx@ltp  #  A'@ltp

    return lt, Delta_ut

# Backward Propagation
def backward_pass(xx, uu, llambdaT):
	"""
	  input: 
				xx state trajectory: x[1],x[2],..., x[T]	i.e. output of each layer of the neural network
				uu input trajectory: u[0],u[1],..., u[T-1]	i.e. weights of the neural network
				llambdaT terminal condition
	  output: 
				llambda costate trajectory
				delta_u costate output, i.e., the loss gradient
	"""
	llambda = np.zeros((T, d))
	llambda[-1] = llambdaT

	Delta_u = np.zeros((T-1, d, d+1))

	for t in reversed(range(T-1)):  #  T-1,T-2,...,1,0
		llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t+1], xx[t], uu[t])

	return Delta_u

# Binary Cross-Entropy w/ logits - Loss Function
def BCEwithLogits(z, y_true):
	if z >= 0:
		bce = z - y_true*z + np.log(1 + np.exp(-z))
		bce_d = 1 / (1 + np.exp(-z)) - y_true			# Derivative version that not cause overflow for z >= 0
	else:
		bce = -y_true*z + np.log(1 + np.exp(z))
		bce_d = np.exp(z) / (1 + np.exp(z)) - y_true	# Derivative version that not cause overflow for z < 0
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

""" Without sampling: the probability distribution of chosen_class is 1/10 """
# n = n_samples//NN # Number of samples per node
# data_point = np.array([train_D[n*i: n*i+n] for i in range(NN)])
# label_point = np.array([train_y[n*i: n*i+n] for i in range(NN)])
""" ---------------------------------------------------------------------- """

""" With pos/neg sampling: the proability distribution of chosen_class is 1/2 """
pos_idx = np.where(train_y == 1)[0][:n_samples//2]
neg_idx = np.where(train_y == 0)[0][:n_samples-n_samples//2]

n = n_samples//NN # Number of samples per node
indices = np.concatenate((pos_idx, neg_idx))
np.random.shuffle(indices)
indices = indices.reshape((NN, n))

data_point = train_D[indices]
label_point = train_y[indices]
""" ------------------------------------------------------------------------ """

print(f"Training label points:\n{label_point}\n")

# Training

d = 28*28           # Number of neurons in each layer. Same numbers for all the layers
stepsize = 1e-4    	# Learning rate
J = np.zeros((MAXITERS, NN))  # Cost

#  U_t : U_0 Initial Weights / Initial Input Trajectory initialized randomly
UU = np.random.randn(MAXITERS+1, NN, T-1, d, d+1)
Delta_u = np.zeros_like(UU) # Weights update

ZZ = np.zeros_like(UU)	# z_t: z_0 Initialized at 0

print(f"Training...")
for tt in range(MAXITERS):  # For each iteration

	for ii in range(NN):  # For each node
		totalCost = 0  # Sum up the cost of each image
		for kk in range(n):  # For each image of the node

			image = data_point[ii][kk]
			label = label_point[ii][kk]

			# Forward pass
			XX = forward_pass(UU[tt, ii], image)  # f_i(x_{i,t})
			# Cost function
			cost, cost_d = BCEwithLogits(XX[-1,-1], label)
			totalCost += cost
			llambdaT = cost_d

			# Backward propagation            # \nabla f_i(x_{i,t})
			Delta_u[tt, ii] += backward_pass(XX, UU[tt, ii], llambdaT) # Sum weigth errors on the full batch of images

		# Store the Loss Value across Iterations (the sum of costs of all nodes)
		J[tt, ii] += totalCost

	for ii in range(NN):  # For each node
		Nii = np.nonzero(Adj[ii])[0] # Self loop is not present

		# Weights update
		UU[tt+1, ii] = WW[ii, ii]*UU[tt, ii]
		for jj in Nii:
			UU[tt+1, ii] += WW[ii, jj]*UU[tt, jj]
		UU[tt+1, ii] += ZZ[tt, ii] - stepsize*Delta_u[tt, ii]

		# ZZ update
		ZZ[tt+1, ii] = WW[ii, ii]*ZZ[tt, ii] - stepsize*(WW[ii, ii]*Delta_u[tt, ii] - Delta_u[tt, ii])
		for jj in Nii:
			ZZ[tt+1, ii] += WW[ii, jj]*ZZ[tt, jj] - stepsize*WW[ii, jj]*Delta_u[tt, jj]

	if (tt % 1) == 0:
		print(f"Iteration {tt:3d} - loss: {np.sum(J[tt]):4.3f}", end="\n")

print(np.sum(np.sum(Delta_u, axis=-1), axis=-1)[:-1, 0, :])

# Plot evolution of cost function
fig = plt.figure()
plt.semilogy(np.arange(MAXITERS), np.sum(J, axis=1), linestyle='-', linewidth=2)
plt.xlabel(r"iterations $t$")
plt.ylabel(r"cost")
plt.title(r"Evolution of the cost error: $\min \sum_{i=1}^N \sum_{k=1}^\mathcal{I} J(\phi(u;x_i^k);y_i^k)$")
plt.grid()
plt.show()
fig.savefig('./Task1/imgs/BCE/Cost error.png')

# Plot single cost of each agent over time
fig = plt.figure()
plt.plot(np.arange(MAXITERS), J)
plt.title(r"Evolution of the cost error (single agents)")
plt.show()
fig.savefig('./Task1/imgs/BCE/Cost error (single agents).png')

# Plot the gradient onf one agent over time for each layer
fig = plt.figure()
plt.plot(np.arange(MAXITERS), np.sum(np.sum(Delta_u, axis=-1), axis=-1)[:-1, 0, :])
plt.title(r"Evolution of the the gradient of one agent")
plt.show()
fig.savefig('./Task1/imgs/BCE/Evolution of the gradient of one agent.png')

# Plot consenus weight for each agent
w = 10
h = 10
fig = plt.figure(figsize=(16, 8))
rows = 1
columns = T-1
ax = []
for layer in range(T-1):
	ax.append(fig.add_subplot(rows, columns, layer+1))
	ax[-1].set_title(f"weights layer: {layer+1}")
	ax[-1].plot(np.arange(MAXITERS), UU[:-1, :, layer, 0, :10].reshape(MAXITERS,-1))
fig.savefig('./Task1/imgs/BCE/Weights Convercenge.png')
plt.show()


# Evaluation on test set
y_pred = []

idx = np.argsort(np.random.random(test_D.shape[0]))
test_images = [test_D[i] for i in idx][:n_samples]
test_labels = [test_y[i] for i in idx][:n_samples]
for image, label in zip(test_images, test_labels):
	# Forward pass
	XX = forward_pass(UU[-1, ii], image)  # f_i(x_i,t)
	y_pred.append(1 if XX[-1, -1] > 0.5 else 0)

print(y_pred)
print(test_labels)

print()

print(f'accuracy: {accuracy_score(test_labels, y_pred):.4f}')
weights = [1 - test_labels.count(1) / len(test_labels) if i == 1 else 1 -
		   test_labels.count(0) / len(test_labels) for i in test_labels]
print(f'accuracy with weights: {accuracy_score(test_labels, y_pred, sample_weight=weights):.4f}')