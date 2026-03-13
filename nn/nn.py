# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # compute linear transformation
        Z_curr = W_curr.dot(A_prev) + b_curr

        # non-linear activation
        if activation.lower() == 'relu': # if relu, apply relu activation
            A_curr = self._relu(Z_curr)
        elif activation.lower() == 'sigmoid': # if sigmoid, apply sigmoid activation
            A_curr = self._sigmoid(Z_curr)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # ensure numpy array and convert to column orientation for internal math
        X = np.array(X)
        A_prev = X.T  # shape: (input_dim, m)

        cache: Dict[str, ArrayLike] = {} # dictionary to store Z and A matrices for backprop
        cache['A0'] = A_prev # store input as A0 for backprop

        A_curr = A_prev # initialize A_curr for first layer as input, will be updated in loop
        for idx, layer in enumerate(self.arch): # iterate through layers
            layer_idx = idx + 1 # create a 1-indexed layer index for parameter retrieval and cache storage
            W_curr = self._param_dict['W' + str(layer_idx)] # retrieve current layer's weight matrix
            b_curr = self._param_dict['b' + str(layer_idx)] # retrieve current layer's bias matrix
            activation = layer.get('activation', 'relu') # get activation function for current layer

            # perform forward pass for current layer and retrieve A and Z matrices
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_curr, activation)

            cache['Z' + str(layer_idx)] = Z_curr # store Z matrix for backprop
            cache['A' + str(layer_idx)] = A_curr # store A matrix for backprop

        # return output in row-wise format (m, output_dim)
        output = A_curr.T
        return output, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        m = A_prev.shape[1] # number of samples in mini-batch (column orientation)

        # compute dZ depending on activation
        if activation_curr.lower() == 'relu': # if relu, apply relu backprop
            dZ = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr.lower() == 'sigmoid': # if sigmoid, apply sigmoid backprop
            dZ = self._sigmoid_backprop(dA_curr, Z_curr)
        else:
            raise ValueError(f"Unsupported activation: {activation_curr}")

        dW_curr = (1.0 / m) * dZ.dot(A_prev.T) # compute weight gradient
        db_curr = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True) # compute bias gradient
        dA_prev = W_curr.T.dot(dZ) # compute gradient with respect to previous layer's activation

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        # prepare y and y_hat as numpy arrays (row-wise)
        y = np.array(y)
        y_hat = np.array(y_hat)

        # determine dA for last layer based on loss function
        loss_name = self._loss_func.lower() # standardize loss function name to lowercase for comparison
        if loss_name in ('binary_cross_entropy', 'bce'): # if bce, apply bce backprop
            dA_last = self._binary_cross_entropy_backprop(y, y_hat)
        elif loss_name in ('mean_squared_error', 'mse'): # if mse, apply mse backprop
            dA_last = self._mean_squared_error_backprop(y, y_hat)
        else:
            raise ValueError(f"Unsupported loss function: {self._loss_func}")

        grad_dict: Dict[str, ArrayLike] = {} # dictionary to store gradients for all layers

        dA_curr = dA_last # initialize dA_curr for last layer as dA from loss function, will be updated in loop
        # iterate through layers backwards
        for idx in reversed(range(len(self.arch))):
            layer_idx = idx + 1 # convert to 1-indexed layer index for parameter retrieval and cache access
            W_curr = self._param_dict['W' + str(layer_idx)] # retrieve current layer's weight matrix
            b_curr = self._param_dict['b' + str(layer_idx)] # retrieve current layer's bias matrix
            Z_curr = cache['Z' + str(layer_idx)] # retrieve current layer's Z matrix from cache
            A_prev = cache['A' + str(layer_idx - 1)] # retrieve previous layer's A matrix from cache
            activation = self.arch[idx].get('activation', 'relu') # retrieve current layer's activation function

            # perform backprop for current layer and retrieve dA for previous layer, and dW and db for current layer
            dA_prev, dW_curr, db_curr = self._single_backprop(
                W_curr, b_curr, Z_curr, A_prev, dA_curr, activation
            )

            grad_dict['dW' + str(layer_idx)] = dW_curr # store dW for current layer in grad_dict
            grad_dict['db' + str(layer_idx)] = db_curr # store db for current layer in grad_dict

            dA_curr = dA_prev # update dA_curr for next iteration of loop (previous layer)

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        # iterate through layers and update parameters using gradient descent
        for idx, _ in enumerate(self.arch):
            layer_idx = idx + 1
            # move weights and biases in direction of negative gradient by step size of learning rate
            self._param_dict['W' + str(layer_idx)] -= self._lr * grad_dict['dW' + str(layer_idx)] # update weight matrix for current layer
            self._param_dict['b' + str(layer_idx)] -= self._lr * grad_dict['db' + str(layer_idx)] # update bias matrix for current layer

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        # ensure args are numpy arrays
        X_train = np.array(X_train) 
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        n_samples = X_train.shape[0] # number of samples in training set (assumes row-wise format)
        per_epoch_loss_train: List[float] = [] # list to store average training loss for each epoch
        per_epoch_loss_val: List[float] = [] # list to store validation loss for each epoch

        for ep in range(self._epochs):
            # shuffle
            perm = np.random.permutation(n_samples)
            X_shuf = X_train[perm]
            y_shuf = y_train[perm]

            epoch_losses: List[float] = [] # list to store loss for each mini-batch in the current epoch

            # mini batch training
            for i in range(0, n_samples, self._batch_size):
                X_batch = X_shuf[i:i + self._batch_size] # get mini batch of training data
                y_batch = y_shuf[i:i + self._batch_size] # get mini batch of training labels

                y_hat_batch, cache = self.forward(X_batch) # perform forward pass on mini batch and retrieve cache for backprop

                # compute loss for this batch
                if self._loss_func.lower() in ('binary_cross_entropy', 'bce'): # if bce, compute bce loss
                    loss = self._binary_cross_entropy(y_batch, y_hat_batch)
                else: 
                    loss = self._mean_squared_error(y_batch, y_hat_batch) # else, compute mse loss

                epoch_losses.append(loss) # record loss for this mini-batch

                # backprop and update
                grad_dict = self.backprop(y_batch, y_hat_batch, cache) # perform backpropagation to compute gradients
                self._update_params(grad_dict) # update parameters using computed gradients

            # record average training loss for epoch
            per_epoch_loss_train.append(float(np.mean(epoch_losses)))

            # validation loss
            y_hat_val, _ = self.forward(X_val) # perform forward pass on validation set
            if self._loss_func.lower() in ('binary_cross_entropy', 'bce'): # if bce, compute bce loss
                val_loss = self._binary_cross_entropy(y_val, y_hat_val)
            else:
                val_loss = self._mean_squared_error(y_val, y_hat_val) # else, compute mse loss

            per_epoch_loss_val.append(float(val_loss)) # record validation loss for epoch

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X) # get predictions from forward pass
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        Z = np.array(Z)
        sigmoid = 1.0 / (1.0 + np.exp(-Z)) # compute sigmoid activation
        return sigmoid

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        s = self._sigmoid(Z) # use sigmoid function to compute activation values for current layer
        dZ = dA * s * (1 - s) # compute derivative of sigmoid for backprop
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z) # compute ReLU activation

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = np.array(dA, copy=True) # copy dA to avoid modifying original
        dZ[Z <= 0] = 0 # apply ReLU derivative
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        eps = 1e-8 # small constant to avoid log(0)

        # ensure y and y_hat are numpy arrays for element-wise operations
        y = np.array(y) 
        y_hat = np.array(y_hat)

        # clip predictions
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)

        m = y.shape[0]
        # compute per-element loss and average over samples and outputs
        loss = - (y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
        return float(np.mean(loss))

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        eps = 1e-8 # small constant to avoid division by zero

        # ensure y and y_hat are numpy arrays for element-wise operations
        y = np.array(y)
        y_hat = np.array(y_hat)

        # clip predictions
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
        m = y.shape[0]

        # compute derivative w.r.t. predictions (row-wise), then convert to column orientation
        dA = (- (y / y_hat_clipped) + ((1 - y) / (1 - y_hat_clipped))) / m
        return dA.T

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        # ensure y and y_hat are numpy arrays for element-wise operations
        y = np.array(y) 
        y_hat = np.array(y_hat)

        # compute mean squared error (mean over samples and outputs)
        loss = np.mean((y - y_hat) ** 2)
        return float(loss)

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # ensure y and y_hat are numpy arrays for element-wise operations
        y = np.array(y)
        y_hat = np.array(y_hat)
        m = y.shape[0]

        # derivative of mean squared error (mean over samples)
        dA = (2.0 / m) * (y_hat - y)
        return dA.T