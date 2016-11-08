"""
This module implements various losses for the network.
You should fill in code into indicated sections.
"""

def HingeLoss(x, y):
  """
  Computes hinge loss and gradient of the loss with the respect to the input for multiclass SVM.

  Args:
    x: Input data.
    y: Labels of data.

  Returns:
    loss: Scalar hinge loss.
    dx: Gradient of the loss with the respect to the input x.

  """
  ########################################################################################
  # TODO:                                                                                #
  # Compute hinge loss on input x and y and store it in loss variable. Compute gradient  #
  # of the loss with respect to the input and store it in dx variable.                   #
  ########################################################################################
  dx = None
  loss = None
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx

def CrossEntropyLoss(x, y):
  """
  Computes cross entropy loss and gradient with the respect to the input.

  Args:
    x: Input data.
    y: Labels of data.

  Returns:
    loss: Scalar cross entropy loss.
    dx: Gradient of the loss with the respect to the input x.

  """
  ########################################################################################
  # TODO:                                                                                #
  # Compute cross entropy loss on input x and y and store it in loss variable. Compute   #
  # gradient of the loss with respect to the input and store it in dx.                   #
  ########################################################################################
  dx = None
  loss = None
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx


def SoftMaxLoss(x, y):
  """
  Computes the loss and gradient with the respect to the input for softmax classfier.

  Args:
    x: Input data.
    y: Labels of data.

  Returns:
    loss: Scalar softmax loss.
    dx: Gradient of the loss with the respect to the input x.

  """
  ########################################################################################
  # TODO:                                                                                #
  # Compute softmax loss on input x and y and store it in loss variable. Compute gradient#
  # of the loss with respect to the input and store it in dx variable.                   #
  ########################################################################################
  softmax = np.exp(x) / np.sum(np.exp(x))

  dx = np.dot(self.params['w'].T, y-softmax)
  loss = np.sum(-np.multiply(np.matrix(y), np.log(softmax)))/np.shape(x)[0]
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx
