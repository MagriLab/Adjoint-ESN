import numpy as np
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs

class ESN:
    def __init__(self, reservoir_size, dimension, connectivity, 
                    input_scaling = 1.0, spectral_radius = 1.0,
                    leak_factor = 1.0, tikhonov = 1e-12, bias_in = 1.0):
        """ Creates an Echo State Network with the given parameters
            Args:
                reservoir_size: number of neurons in the reservoir
                dimension: dimension of the state space of the input and output #@todo: separate input and output dimensions
                connectivity: connectivity of the reservoir weights, how many connections does each neuron have (on average)
                input_scaling: scaling applied to the input weights matrix
                spectral_radius: spectral radius (maximum absolute eigenvalue) of the reservoir weights matrix
                leak_factor: factor for the leaky integrator if set to 1 (default), then no leak is applied
                tikhonov: regularisation coefficient for the ridge regression
            Returns:
                ESN object

        """
        # global parameters
        self.N_reservoir = reservoir_size
        self.N_dim = dimension
        self.connectivity = connectivity

        self.sparseness = 1-(self.connectivity/(self.N_reservoir-1))
        # probability of non-connections = 1 - probability of connection
        # probability of connection = (number of connections)/(total number of neurons - 1)
        # -1 to exclude the neuron itself
    
        self.alpha = leak_factor
        self.tikh = tikhonov

        self.sigma_in = input_scaling
        self.rho = spectral_radius

        # weights
        self.W_in = self.generate_input_weights()
        self.W = self.generate_reservoir_weights()
        self.W_out = np.zeros((self.N_reservoir, self.N_dim))

        # biases
        self.bias_in = bias_in
        self.bias_out = np.array([1.0])

    def generate_input_weights(self, seeds = [0,1]):
        """ Create the input weights matrix
        Args:
            seeds: a list of seeds for the random generators; 
                one for the column index, one for the uniform sampling
        Returns:
            W_in: sparse matrix containing the input weights
        """
        # initialize W_in with zeros
        W_in = lil_matrix((self.N_reservoir, self.N_dim+1)) # N_dim+1 because we augment the inputs with a bias

        # set the seeds
        rnd0 = np.random.RandomState(seeds[0])
        rnd1 = np.random.RandomState(seeds[1])

        # make W_in
        for j in range(self.N_reservoir):
            rnd_idx = rnd0.randint(0,self.N_dim+1)
            # only one element different from zero
            W_in[j,rnd_idx] = rnd1.uniform(-1,1) # sample from the uniform distribution
        W_in = W_in.tocsr()

        # scale W_in
        W_in = self.sigma_in*W_in

        return W_in

    def generate_reservoir_weights(self, seeds = [2,3]):
        """ Create the reservoir weights matrix according to Erdos-Renyi network
        Args:
            seeds: a list of seeds for the random generators; 
                one for the connections, one for the uniform sampling of weights
        Returns:
            W: sparse matrix containing reservoir weights
        """
        # set the seeds
        rnd0 = np.random.RandomState(seeds[0]) # connection rng
        rnd1 = np.random.RandomState(seeds[1])  # sampling rng

        # initialize with zeros
        W = np.zeros((self.N_reservoir, self.N_reservoir))

        # generate a matrix sampled from the uniform distribution (0,1)
        W_connection = rnd0.rand(self.N_reservoir, self.N_reservoir) 

        # generate the weights from the uniform distribution (-1,1)
        W_weights = rnd1.uniform(-1,1,(self.N_reservoir, self.N_reservoir)) 

        # replace the connections with the weights
        W = np.where(W_connection<(1-self.sparseness), W_weights, W)
        # 1-sparseness is the connection probability = p, 
        # after sampling from the uniform distribution between (0,1), 
        # the probability of being in the region (0,p) is the same as having probability p
        # (this is equivalent to drawing from a Bernoulli distribution with probability p)

        W = csr_matrix(W)

        # find the spectral radius of the generated matrix
        # this is the maximum absolute eigenvalue
        rho_pre = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]

        # first scale W by the spectral radius to get unitary spectral radius
        W = (1/rho_pre)*W 

        # scale again with the user specified spectral radius
        W = self.rho*W

        return W

    def step(self, x_prev, u, scale = (0,1)):
        """ Advances ESN time step.
        Args:
            x_prev: reservoir state in the previous time step (n-1)
            u: input in this time step (n)
            scale: tuple that contains the scaling parameters for input
        Returns:
            x_next: reservoir state in this time step (n)
        """
        # normalise the input
        u_norm = (u-scale[0])/scale[1] # we normalize here, so that the input is normalised in closed-loop run too?

        # augment the input with the input bias
        u_augmented = np.hstack((u_norm, self.bias_in))

        # update the reservoir
        x_tilde = np.tanh(self.W_in.dot(u_augmented)+ self.W.dot(x_prev))

        # apply the leaky integrator
        x = (1-self.alpha)*x_prev+self.alpha*x_tilde
        return x

    def open_loop(self, x0, U, scale = (0,1)):
        """ Advances ESN in open-loop.
            Args:
                x0: initial reservoir state
                U: input time series in matrix form (N_t x N_dim)
                scale: tuple that contains the scaling parameters for input
            Returns:
                X: time series of the reservoir states (N_t x N_reservoir)
        """
        N_t = U.shape[0] # number of time steps

        # create an empty matrix to hold the reservoir states in time
        X = np.empty((N_t+1, self.N_reservoir)) # N_t+1 because at t = 0, we don't have input
        
        # initialise with the given initial reservoir states
        X[0,:] = x0

        # step in time 
        for n in np.arange(1, N_t+1):
            X[n] = self.step(X[n-1,:], U[n-1,:], scale) # update the reservoir
        
        return X

    def closed_loop(self, x0, N_t, scale = (0,1)):
        """ Advances ESN in closed-loop.
            Args:
                N_t: number of time steps
                x0: initial reservoir state
                scale: tuple that contains the scaling parameters for input
            Returns:
                X: time series of the reservoir states (N_t x N_reservoir)
                Y: time series of the output (N_t x N_dim)
        """
        # create an empty matrix to hold the reservoir states in time
        X = np.empty((N_t+1, self.N_reservoir)) 

        # create an empty matrix to hold the output states in time
        Y = np.empty((N_t+1, self.N_dim)) 

        # initialize with the given initial reservoir states
        X[0,:] = x0

        # augment the reservoir states with the bias
        x0_augmented = np.hstack((x0, self.bias_out))
        # initialise with the calculated output states
        Y[0,:] = np.dot(x0_augmented, self.W_out) 

        # step in time
        for n in range(1, N_t+1):
            X[n,:] = self.step(X[n-1,:], Y[n-1,:], scale) # update the reservoir with the feedback from the output
            x_augmented = np.hstack((X[n,:], self.bias_out)) # augment the reservoir states with bias
            Y[n,:] = np.dot(x_augmented, self.W_out) # update the output with the reservoir states
        
        return X, Y
        
    def train(self, U_washout, U_train, Y_train, scale = (0,1)):
        """ Trains ESN and sets the output weights.
            Args:
                U_washout: washout input time series
                U_train: training input time series
                Y_train: training output time series
                scale: tuple that contains the scaling parameters for input
        """
        # Wash-out phase to get rid of the effects of reservoir states initialised as zero
        x0_washout = np.zeros(self.N_reservoir) # initialise the reservoir states before washout

        # let the ESN run in open-loop for the wash-out
        # get the initial reservoir for the training, which is the last reservoir state
        x0_train = self.open_loop(x0=x0_washout, U=U_washout, scale = scale)[-1,:] 

        # run in open loop with the training data
        X_train = self.open_loop(x0=x0_train, U=U_train, scale = scale) 
        # X_train is one step longer than U_train and Y_train, we discard the initial state
        X_train = X_train[1:,:]

        # augment with the bias
        N_t = X_train.shape[0] # number of time steps
        X_train_augmented = np.hstack((X_train, self.bias_out*np.ones((N_t,1))))
        
        # solve for W_out using ridge regression
        # @todo:
        # can set the method for ridge regression, compare the methods
        # scikit recommends minibatch sgd method for large scale data
        # Alberto implements the closed-form solution because he doesn't want to recalculate
        # the matmuls for each tikhonov parameter?
        reg = Ridge(alpha=self.tikh, fit_intercept=False)
        reg.fit(X_train_augmented, Y_train)
        self.W_out = reg.coef_.T

        return
        










