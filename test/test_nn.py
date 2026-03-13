import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs


def test_single_forward():
    """
    Validates single layer forward pass with correct output shapes and activation function behavior.
    Checks that ReLU activation outputs are non-negative and that the linear transformation is correct.
    Checks that the output of the forward pass has the expected values for a simple input and known weights/biases.
    """
    # create a simple neural network
    nn_arch = [{'input_dim': 2, 'output_dim': 3, 'activation': 'relu'}]
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=4, epochs=10, loss_function='mse')
    
    # create input: (3, 2) where 3 is batch size, 2 is input dimension
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    # get weights and biases
    W = model._param_dict['W1']
    b = model._param_dict['b1']
    
    # compute single forward pass
    A_prev = X.T  # (2, 3)
    A_curr, Z_curr = model._single_forward(W, b, A_prev, 'relu')
    
    # check shapes
    assert A_curr.shape == (3, 3), f"Expected shape (3, 3), got {A_curr.shape}"
    assert Z_curr.shape == (3, 3), f"Expected shape (3, 3), got {Z_curr.shape}"
    
    # ReLU output should be >= 0
    assert np.all(A_curr >= 0), "ReLU output should be non-negative"

    # manual calculation for expected Z and A
    expected_Z = W.dot(A_prev) + b
    expected_A = np.maximum(0, expected_Z)  # ReLU activation
    np.testing.assert_array_almost_equal(Z_curr, expected_Z, decimal=5, err_msg="Z_curr does not match expected values")    
    np.testing.assert_array_almost_equal(A_curr, expected_A, decimal=5, err_msg="A_curr does not match expected values")

def test_forward():
    """
    Test full forward pass through network.
    Validates that output has correct shape and values are in expected range for sigmoid activation.
    Checks that cache contains correct intermediate values for all layers.
    """
    nn_arch = [
        {'input_dim': 4, 'output_dim': 3, 'activation': 'relu'},
        {'input_dim': 3, 'output_dim': 2, 'activation': 'sigmoid'}
    ]
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=4, epochs=10, loss_function='bce')
    
    # create input: (5, 4) where 5 is batch size, 4 is input dimension
    X = np.random.randn(5, 4)
    
    output, cache = model.forward(X) # perform forward pass and get output and cache for all layers
    
    # check output shape: (batch_size, output_dim) = (5, 2)
    assert output.shape == (5, 2), f"Expected shape (5, 2), got {output.shape}"
    
    # check cache contains all Z and A matrices
    assert 'A0' in cache, "Cache should contain A0 (input)"
    assert 'Z1' in cache, "Cache should contain Z1"
    assert 'A1' in cache, "Cache should contain A1"
    assert 'Z2' in cache, "Cache should contain Z2"
    assert 'A2' in cache, "Cache should contain A2 (output)"
    
    # sigmoid output should be in (0, 1)
    assert np.all(output > 0) and np.all(output < 1), "Sigmoid output should be in (0, 1)"

    W1 = model._param_dict['W1']
    b1 = model._param_dict['b1']
    W2 = model._param_dict['W2']
    b2 = model._param_dict['b2']

    # manual calculation for expected output
    A0 = X.T  # (4, 5)
    Z1 = W1.dot(A0) + b1
    A1 = np.maximum(0, Z1)  # ReLU activation
    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation
    expected_output = 1 / (1 + np.exp(-Z2))  # Sigmoid activation
    np.testing.assert_allclose(A0, cache['A0'], rtol=1e-5, err_msg="A0 in cache does not match expected input")
    np.testing.assert_allclose(Z1, cache['Z1'], rtol=1e-5, err_msg="Z1 in cache does not match expected values")
    np.testing.assert_allclose(A1, cache['A1'], rtol=1e-5, err_msg="A1 in cache does not match expected values")
    np.testing.assert_allclose(Z2, cache['Z2'], rtol=1e-5, err_msg="Z2 in cache does not match expected values")
    np.testing.assert_allclose(A2, cache['A2'], rtol=1e-5, err_msg="A2 in cache does not match expected output")
    

def test_single_backprop():
    """
    Test single layer backprop.
    Validates that backprop returns gradients of correct shape and that values are finite.
    Checks that the computed gradients match expected values for a simple input and known weights/biases"""
    nn_arch = [{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'}]
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=4, epochs=10, loss_function='mse')
    
    # setup
    m = 4  # batch size
    W = model._param_dict['W1']  # (2, 3)
    b = model._param_dict['b1']  # (2, 1)
    A_prev = np.random.randn(3, m)
    Z_curr = W.dot(A_prev) + b
    A_curr = model._relu(Z_curr)
    
    # gradient from next layer
    dA_curr = np.random.randn(2, m)
    
    # backprop
    dA_prev, dW_curr, db_curr = model._single_backprop(W, b, Z_curr, A_prev, dA_curr, 'relu')
    
    # check shapes
    assert dA_prev.shape == A_prev.shape, f"dA_prev shape mismatch: {dA_prev.shape} vs {A_prev.shape}"
    assert dW_curr.shape == W.shape, f"dW_curr shape mismatch: {dW_curr.shape} vs {W.shape}"
    assert db_curr.shape == b.shape, f"db_curr shape mismatch: {db_curr.shape} vs {b.shape}"
    
    # gradients should be finite
    assert np.all(np.isfinite(dA_prev)), "dA_prev contains non-finite values"
    assert np.all(np.isfinite(dW_curr)), "dW_curr contains non-finite values"
    assert np.all(np.isfinite(db_curr)), "db_curr contains non-finite values"

    # manual backprop calculation for expected gradients
    dZ_curr = dA_curr * (Z_curr > 0)  # ReLU backprop
    expected_dW = (1/m) * dZ_curr.dot(A_prev.T)
    expected_db = (1/m) * np.sum(dZ_curr, axis=1, keepdims=True)
    expected_dA_prev = W.T.dot(dZ_curr)
    np.testing.assert_allclose(dW_curr, expected_dW, rtol=1e-5, err_msg="dW_curr does not match expected values")
    np.testing.assert_allclose(db_curr, expected_db, rtol=1e-5, err_msg="db_curr does not match expected values")
    np.testing.assert_allclose(dA_prev, expected_dA_prev, rtol=1e-5, err_msg="dA_prev does not match expected values")
    

def test_predict():
    """
    Test prediction method.
    Validates that predict returns output of correct shape 
    Checks if values are in expected range for sigmoid activation.
    """
    nn_arch = [
        {'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
        {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
    ]
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=4, epochs=10, loss_function='bce')
    
    X = np.random.randn(5, 3) # 5 samples, 3 features
    predictions = model.predict(X) # get predictions for input X
    
    # check shape
    assert predictions.shape == (5, 1), f"Expected shape (5, 1), got {predictions.shape}"
    
    # predictions should be in valid range for sigmoid
    assert np.all(predictions > 0) and np.all(predictions < 1), "Predictions should be in (0, 1)"

    # manual calculation for expected predictions
    exprected_predictions = model.forward(X)[0]  # get output from forward pass
    np.testing.assert_allclose(predictions, exprected_predictions, rtol=1e-5, err_msg="Predictions do not match expected values from forward pass") 
    

def test_binary_cross_entropy():
    """
    Test binary cross entropy loss function.
    Validates that BCE loss is non-negative, lower for better predictions
    Check if correctly computed for close predictions.
    """
    nn_arch = [{'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=4, epochs=10, loss_function='bce')
    
    # test case 1: perfect predictions
    y = np.array([[1.0], [0.0], [1.0]])
    y_hat = np.array([[0.99], [0.01], [0.99]])
    loss = model._binary_cross_entropy(y, y_hat)
    
    assert isinstance(loss, float), "Loss should be a float"
    assert loss >= 0, "BCE loss should be non-negative"
    assert loss < 1, "Perfect predictions should have low loss"
    
    # test case 2: worse predictions
    y_hat_bad = np.array([[0.1], [0.9], [0.1]])
    loss_bad = model._binary_cross_entropy(y, y_hat_bad)
    
    assert loss_bad > loss, "Worse predictions should have higher loss"

    # test case 3: manual calculation for expected loss
    epsilon = 1e-15
    y_hat_clipped = np.clip(y_hat, epsilon, 1 - epsilon)
    expected_loss = -np.mean(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
    assert np.isclose(loss, expected_loss, rtol=1e-5), "Computed loss does not match expected values"
    

def test_binary_cross_entropy_backprop():
    """
    Test binary cross entropy backprop.
    Validates that BCE backprop returns gradients of correct shape and that values are finite.
    Checks that the computed gradients match expected values for a simple input and known predictions.
    """
    nn_arch = [{'input_dim': 3, 'output_dim': 2, 'activation': 'sigmoid'}]
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=4, epochs=10, loss_function='bce')
    
    # setup simple case with known values
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_hat = np.array([[0.8, 0.2], [0.3, 0.7]])
    
    # call backprop
    dA = model._binary_cross_entropy_backprop(y, y_hat)
    
    # check shape (should be transposed to column format)
    assert dA.shape == (2, 2), f"Expected shape (2, 2), got {dA.shape}"
    
    # gradients should be finite
    assert np.all(np.isfinite(dA)), "Gradients contain non-finite values"

    # manual calculation for expected gradients
    epsilon = 1e-15
    y_hat_clipped = np.clip(y_hat, epsilon, 1 - epsilon)
    expected_dA = - (y / y_hat_clipped) + ((1 - y) / (1 - y_hat_clipped))
    expected_dA = expected_dA.T / y.shape[0]  # transpose to match shape and average over batch
    np.testing.assert_allclose(dA, expected_dA, rtol=1e-5, err_msg="Computed gradients do not match expected values")
    

def test_mean_squared_error():
    """
    Test mean squared error loss function.
    Validates that MSE loss is non-negative, lower for better predictions
    Check if correctly computed for close predictions."""
    nn_arch = [{'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=4, epochs=10, loss_function='mse')
    
    # test case 1: perfect predictions
    y = np.array([[1.0], [0.0], [0.5]])
    y_hat = np.array([[1.0], [0.0], [0.5]])
    loss = model._mean_squared_error(y, y_hat)
    
    assert loss == 0.0, "Perfect predictions should have zero MSE"
    
    # test case 2: predictions with error
    y_hat_off = np.array([[0.9], [0.1], [0.4]])
    loss_off = model._mean_squared_error(y, y_hat_off)
    
    assert loss_off > 0, "Predictions with error should have positive MSE"
    assert loss_off < 1, "MSE for small errors should be < 1"

    # test case 3: manual calculation for expected loss
    expected_loss = np.mean((y - y_hat) ** 2)
    assert np.isclose(loss, expected_loss, rtol=1e-5), "Computed loss does not match expected values"       
    

def test_mean_squared_error_backprop():
    """
    Test mean squared error backprop.
    Validates that MSE backprop returns gradients of correct shape and that values are finite.
    Checks that the computed gradients match expected values for a simple input and known predictions.
    """
    nn_arch = [{'input_dim': 3, 'output_dim': 2, 'activation': 'sigmoid'}]
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=4, epochs=10, loss_function='mse')
    
    # setup simple case with known values
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_hat = np.array([[0.8, 0.2], [0.3, 0.7]])
    
    # call backprop
    dA = model._mean_squared_error_backprop(y, y_hat)
    
    # check shape (should be transposed to column format)
    assert dA.shape == (2, 2), f"Expected shape (2, 2), got {dA.shape}"
    
    # gradients should be finite
    assert np.all(np.isfinite(dA)), "Gradients contain non-finite values"

    # manual calculation for expected gradients
    expected_dA = (2.0 * (y_hat - y) / y.shape[0]).T  # transpose to match shape and average over batch
    np.testing.assert_allclose(dA, expected_dA, rtol=1e-5, err_msg="Computed gradients do not match expected values")


def test_sample_seqs():
    """
    Test sequence sampling with replacement.
    Validates that sampled sequences and labels have correct lengths and that classes are balanced.
    Checks that all sampled sequences are from the original input.
    """
    # setup simple case with 2 positive and 4 negative examples
    seqs = ['ATCG', 'GCTA', 'TTAA', 'GGCC', 'AATT', 'CCGG']
    labels = [True, True, False, False, False, False]  
    
    # call sampling function
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)
    
    # should balance to 4 positive and 4 negative (max of 2 and 4)
    assert len(sampled_seqs) == 8, f"Expected 8 sequences, got {len(sampled_seqs)}"
    assert len(sampled_labels) == 8, f"Expected 8 labels, got {len(sampled_labels)}"
    
    # check balance
    n_positive = sum(sampled_labels)
    n_negative = len(sampled_labels) - n_positive
    assert n_positive == 4, f"Expected 4 positive labels, got {n_positive}"
    assert n_negative == 4, f"Expected 4 negative labels, got {n_negative}"
    
    # all sampled sequences should be from original
    for seq in sampled_seqs:
        assert seq in seqs, f"Sampled sequence '{seq}' not in original sequences"


def test_one_hot_encode_seqs():
    """
    Test one-hot encoding of sequences.
    Validates that encoded sequences have correct shape and values are valid one-hot encodings.
    Checks that sequences of different lengths raise an error and that empty input returns empty array.
    """
    # setup simple case with 3 sequences of length 2
    seqs = ['AT', 'GC', 'AA']
    encodings = one_hot_encode_seqs(seqs)
    
    # check shape: (3 sequences, 4 bases * 2 length) = (3, 8)
    assert encodings.shape == (3, 8), f"Expected shape (3, 8), got {encodings.shape}"
    
    # expected encodings:
    # A -> [1,0,0,0], T -> [0,1,0,0], so AT -> [1,0,0,0,0,1,0,0]
    expected_at = np.array([1,0,0,0,0,1,0,0], dtype=float)
    np.testing.assert_array_equal(encodings[0], expected_at)
    
    # G -> [0,0,0,1], C -> [0,0,1,0], so GC -> [0,0,0,1,0,0,1,0]
    expected_gc = np.array([0,0,0,1,0,0,1,0], dtype=float)
    np.testing.assert_array_equal(encodings[1], expected_gc)
    
    # AA -> [1,0,0,0,1,0,0,0]
    expected_aa = np.array([1,0,0,0,1,0,0,0], dtype=float)
    np.testing.assert_array_equal(encodings[2], expected_aa)
    
    # all values should be 0 or 1 (valid one-hot encoding)
    assert np.all((encodings == 0) | (encodings == 1)), "Encodings should only contain 0 or 1"
    
    # each position should sum to 1 (valid one-hot)
    row_sums = encodings.sum(axis=1)
    assert np.all(row_sums == 8), "Each encoding should have exactly 8 ones (4 bases * 2 length)"