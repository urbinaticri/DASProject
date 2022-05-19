from cmath import log
import numpy as np


import matplotlib.pyplot as plt
from keras.datasets import mnist

###############################################################################
# Useful constants
MAXITERS = int(1e4) # Explicit Casting
NN = 10

###############################################################################
# Generate Network Binomial Graph
p_ER = 0.3
I_NN = np.eye(NN) # np.identity
while 1:
	Adj = np.random.binomial(1,p_ER, (NN,NN))
	Adj = np.logical_or(Adj,Adj.T)
	Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)

	test = np.linalg.matrix_power(I_NN+Adj, NN)
	if np.all(test>0):
		break

###############################################################################
# Compute mixing matrix
threshold = 1e-10
WW = 1.5*I_NN + 0.5*Adj

ONES = np.ones((NN,NN))
ZEROS = np.zeros((NN,NN))
WW = np.maximum(WW,0*ONES)
while any(abs(np.sum(WW,axis=1)-1) > threshold):
	WW = WW/(WW@ONES) # row
	# WW = WW/(ONES@WW) # col
	WW = np.maximum(WW,0*ONES)

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
	xx = np.zeros((T,d))
	xx[0] = x0

	for t	in range(T-1):
		xx[t+1] = inference_dynamics(xx[t],uu[t]) # x^+ = f(x,u)

	return xx

# Adjoint dynamics: 
#   state:    lambda_t = A.T lambda_tp
#   output:   deltau_t = B.T lambda_tp
def adjoint_dynamics(ltp,xt,ut):
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

  for t in reversed(range(T-1)): # T-2,T-1,...,1,0
    llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t+1],xx[t],uu[t])

  return Delta_u


###############################################################################
# GO!
# stepsize = 1e-3 # Constant Stepsize

chosen_class = 4
n_samples = 100

(train_D, train_y), (test_D, test_y) = mnist.load_data()
train_D , test_D = train_D/255.0 , test_D/255.0
train_D = train_D.reshape((60000, 28 * 28))

train_y = [1 if y == chosen_class else 0 for y in train_y]
test_y = [1 if y == chosen_class else 0 for y in test_y]

train_D, train_y  = train_D[:n_samples], train_y[:n_samples]
idx  = np.argsort(np.random.random(n_samples))
train_D = [train_D[i] for i in idx]
train_y = [train_y[i] for i in idx]

# Training Set
#label_point = train_y[:NN]
#data_point = train_D[:NN]
label_point = [ train_y[NN*i: NN*i+NN] for i in range(int(n_samples/NN)) ]
data_point  = [ train_D[NN*i: NN*i+NN] for i in range(int(n_samples/NN)) ]


T = 3	# Layers
d = int(n_samples/NN)*28*28	# Number of neurons in each layer. Same numbers for all the layers

# Gradient Method Parameters
max_iters = 10 # epochs
stepsize = 0.1 # learning rate

J = np.zeros((NN, MAXITERS)) # Cost


# Initial Weights / Initial Input Trajectory
#uu = np.random.randn(T-1, d, d+1)
XX = np.random.randn(NN, T-1, d, d+1)
XXtp = np.zeros_like(XX)

for tt in range (MAXITERS-1):


	if (tt % 10) == 0:
		print("Iteration {:3d}".format(tt), end="\n")
	

	for ii in range (NN):

		Nii = np.nonzero(Adj[ii])[0]

		# Initial State Trajectory
		xx = forward_pass(data_point[ii],XX[ii]) # T x d

		# Backward propagation
		llambdaT = 2*( xx[-1,:] - label_point[ii]) # xT
		Delta_u = backward_pass(xx,XX[ii],llambdaT) # the gradient of the loss function

		# Update the weights
		XXtp[ii] = WW[ii,ii]*XX[ii] - stepsize*Delta_u # overwriting the old value
		#XXtp[ii] = WW[ii,ii]*XX[ii] - stepsize*Delta_u/(tt+1)*10 #diminishing
		for jj in Nii:
			XXtp[ii] += WW[ii,jj]*XX[jj]

		# Store the Loss Value across Iterations
		J[ii, tt] = llambdaT
		#BCE = label_point[ii] * log(xx[-1,:]) + (1 - label_point[ii]) * (1 - log(xx[-1,:]))
		#J[ii, tt] = BCE

	XX = XXtp