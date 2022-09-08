import tensorflow as tf 


def mse(y_true, y_pred):
    """ Mean squared error """
    return (y_true - y_pred) ** 2


def msle(y_true, y_pred):
    """ Mean squared logarithmic error """
    return (tf.math.log(y_true + 1) - tf.math.log(y_pred + 1)) ** 2



class SparseTargetLoss(tf.losses.Loss):
    """ Handles target which has only some pixels available """

    def __init__(self, loss_function=mse):
        self.loss = loss_function
        super().__init__()


    def call(self, y_true, y_pred, *args, **kwargs):
        valid = tf.math.is_finite(y_true)
        loss  = self.loss(y_true, y_pred)
        return tf.reduce_sum( tf.where(valid, loss, 0) ) / tf.reduce_sum(valid)