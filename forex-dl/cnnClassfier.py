import tensorflow as tf


class ConvolutionalBatchNormalizer(object):
  """     
  Use:                                                                      
      ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
      bn = ConvolutionalBatchNormalizer(depth, 0.001, ewma, True)           
      update_assignments = bn.get_assigner()                                
      x = bn.normalize(y, train=training?)                                  
      (the output x will be batch-normalized).                              
  """

  def __init__(self, depth, epsilon, ewma_trainer, scale_after_norm):
    self.mean = tf.Variable(tf.constant(0.0, shape=[depth]),
                            trainable=False)
    self.variance = tf.Variable(tf.constant(1.0, shape=[depth]),
                                trainable=False)
    self.beta = tf.Variable(tf.constant(0.0, shape=[depth]), name = 'beta')
    self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]),name = 'gamma')
    self.ewma_trainer = ewma_trainer
    self.epsilon = epsilon
    self.scale_after_norm = scale_after_norm

  def get_assigner(self):
    """Returns an EWMA apply op that must be invoked after optimization."""
    return self.ewma_trainer.apply([self.mean, self.variance])

  def normalize(self, x, train=True):
    """Returns a batch-normalized version of x."""
    mean, variance = tf.nn.moments(x, [0,1,2])
    assign_mean = self.mean.assign(mean)
    assign_variance = self.variance.assign(variance)
    with tf.control_dependencies([assign_mean, assign_variance]):
      return tf.nn.batch_norm_with_global_normalization(
          x, mean, variance, self.beta, self.gamma,
          self.epsilon, self.scale_after_norm)