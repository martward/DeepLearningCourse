import numpy as np
"""
This module implements various layers for the network.
You should fill in code into indicated sections.
"""

class Layer(object):
  """
  Base class for all layers classes.

  """
  def __init__(self, layer_params = None):
    """
    Initializes the layer according to layer parameters.

    Args:
      layer_params: Dictionary with parameters for the layer.

    """
    self.train_mode = False

  def initialize(self):
    """
    Cleans cache. Cache stores intermediate variables needed for backward computation.

    """
    self.cache = None

  def layer_loss(self):
    """
    Returns partial loss of layer parameters for regularization term of full loss.

    Returns:
      loss: Partial loss of layer parameters.

    """
    return 0.

  def set_train_mode(self):
    """
    Sets train mode for the layer.

    """
    self.train_mode = True

  def set_test_mode(self):
    """
    Sets test mode for the layer.

    """
    self.train_mode = False

  def forward(self, X):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    raise NotImplementedError("Forward pass is not implemented for base Layer class.")

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradient of the output with respect to the input of the layer.

    """
    raise NotImplementedError("Backward pass is not implemented for base Layer class.")

class LinearLayer(Layer):
  """
  Linear layer.

  """
  def __init__(self, layer_params):
    """
    Initializes the layer according to layer parameters.

    Args:
      layer_params: Dictionary with parameters for the layer:
          input_size - input dimension;
          output_size - output dimension;
          weight_decay - regularization parameter for the weights;
          weight_scale - scale of normal distrubtion to initialize weights.

    """

    self.layer_params = layer_params
    self.layer_params.setdefault('weight_decay', 0.0)
    self.layer_params.setdefault('weight_scale', 0.001)

    self.params = {'w': None, 'b': None}
    self.grads = {'w': None, 'b': None}

    self.train_mode = False

  def initialize(self):
    """
    Initializes weights and biases. Cleans cache.
    Cache stores intermediate variables needed for backward computation.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Initialize weights self.params['w'] using normal distribution with mean = 0 and      #
    # std = self.layer_params['weight_scale'].                                             #
    #                                                                                      #
    # Initialize biases self.params['b'] with 0.                                           #
    ########################################################################################
    self.params['w'] = np.random.normal(0, self.layer_params['weight_scale'],[self.layer_params['input_size'],self.layer_params['output_size']])
    self.params['b'] = np.zeros(self.layer_params['output_size'])
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    self.cache = None

  def layer_loss(self):
    """
    Returns partial loss of layer parameters for regularization term of full loss.

    Returns:
      loss: Partial loss of layer parameters.

    """

    ########################################################################################
    # TODO:                                                                                #
    # Compute the loss of the layer which responsible for L2 regularization term. Store it #
    # in loss variable.                                                                    #
    ########################################################################################
    loss = self.layer_params['weight_decay'] * np.sum(np.power(self.params['w'],2))
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    return loss

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for LinearLayer. Store output of the layer in out varible.    #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = np.dot(x, self.params['w']) + self.params['b']

    # Cache if in train mode
    if self.train_mode:
      self.cache = x
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    if not self.train_mode:
      raise ValueError("Backward is not possible in test mode")

    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for LinearLayer. Store gradient of the loss with respect to  #
    # layer parameters in self.grads['w'] and self.grads['b']. Store gradient of the loss  #
    # with respect to the input in dx variable.                                            #
    #                                             s                                         #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################
    n = dout.shape[0]
    dx = np.dot(self.params['w'], dout.T)
    self.grads['w'] = (np.dot(self.cache.T, dout) / n) + self.layer_params['weight_decay'] * self.params['w']
    self.grads['b'] = (np.sum(dout,0) / n) + self.layer_params['weight_decay'] * self.params['b']
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx.T

class ReLULayer(Layer):
  """
  ReLU activation layer.

  """

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for ReLULayer. Store output of the layer in out variable.     #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = x.clip(min=0)
    # Cache if in train mode
    if self.train_mode:
      self.cache = x
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for ReLULayer. Store gradient of the loss with respect to    #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################
    xReLU = self.cache.clip(min=0)
    dx = xReLU
    dx[dx > 0 ] = 1
    dx = np.multiply(dx, dout)
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    return dx

class SigmoidLayer(Layer):
  """
  Sigmoid activation layer.

  """
  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for SigmoidLayer. Store output of the layer in out variable.  #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = np.divide(1.0, 1+np.exp(-x))

    # Cache if in train mode
    if self.train_mode:
      self.cache = x
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for SigmoidLayer. Store gradient of the loss with respect to #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################
    S = np.divide(1.0, 1+np.exp(-self.cache))
    S2 = np.multiply(S,S)
    dx = np.multiply(S - S2,dout)
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class TanhLayer(Layer):
  """
  Tanh activation layer.

  """
  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for TanhLayer. Store output of the layer in out variable.     #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = np.tanh(x)

    # Cache if in train mode
    if self.train_mode:
      self.cache = x
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for TanhLayer. Store gradient of the loss with respect to    #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################

    dx = np.multiply(1 - np.power(np.tanh(self.cache),2),dout)
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class ELULayer(Layer):
  """
  ELU activation layer.

  """

  def __init__(self, layer_params):
    """
    Initializes the layer according to layer parameters.

    Args:
      layer_params: Dictionary with parameters for the layer:
          alpha - alpha parameter;

    """
    self.layer_params = layer_params
    self.layer_params.setdefault('alpha', 1.0)
    self.train_mode = False

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for ELULayer. Store output of the layer in out variable.      #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################

    z = np.ones(x.shape)
    z[x < 0] = (z[x<0]  * self.layer_params['alpha'] )- self.layer_params['alpha']
    out = np.multiply(x,z)

    # Cache if in train mode
    if self.train_mode:
      self.cache = x
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for ELULayer. Store gradient of the loss with respect to     #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################

    z = np.exp(self.cache) * self.layer_params['alpha']
    z[self.cache > 0] = 1
    dx = np.multiply(z,dout)
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class SoftMaxLayer(Layer):
  """
  Softmax activation layer.

  """

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for SoftMaxLayer. Store output of the layer in out variable.  #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = np.exp(x) / np.sum(np.exp(x), 1)

    # Cache if in train mode
    if self.train_mode:
      self.cache = x
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for SoftMaxLayer. Store gradient of the loss with respect to #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################s
    # calculate diagonal values
    dx = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx
