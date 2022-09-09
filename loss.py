import tensorflow as tf 


def mse(y_true, y_pred):
    """ Mean squared error """
    return tf.reduce_mean((y_true - y_pred) ** 2)


def msle(y_true, y_pred):
    """ Mean squared logarithmic error """
    return tf.reduce_mean((tf.math.log(y_true + 1) - tf.math.log(y_pred + 1)) ** 2)



class SparseTargetLoss(tf.losses.Loss):
    """ Handles target which has only some pixels available """

    def __init__(self, loss_function=mse):
        self.loss = loss_function
        super().__init__()


    def call(self, y_true, y_pred, *args, **kwargs):
        """ Discard any entries which are NaN in y_true """
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        valid  = tf.where( tf.math.is_finite(y_true) )
        y_true = tf.gather(y_true, valid)
        y_pred = tf.gather(y_pred, valid)
        return self.loss(y_true, y_pred)