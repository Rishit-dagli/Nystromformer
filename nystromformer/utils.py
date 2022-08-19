import tensorflow as tf
from einops import rearrange, reduce

class MoorePenrosePseudoinverse(tf.keras.layers.Layer):
    def __init__(self, iteration=6, **kwargs):
        super(MoorePenrosePseudoinverse, self).__init__(**kwargs)

        self.iteration = iteration

    def call(self, inputs, **kwargs):
        abs_inputs = tf.abs(inputs)
        cols = tf.math.reduce_sum(abs_inputs, axis=-1)
        rows = tf.math.reduce_sum(abs_inputs, axis=-2)
        z = rearrange(inputs, '... i j -> ... j i') / (tf.math.reduce_max(cols) * tf.math.reduce_max(rows))

        identity = tf.eye(z.shape[-1])
        identity = rearrange(identity, 'i j -> () i j')

        for _ in range(self.iteration):
            inputs_bbm_z = inputs @ z
            z = 0.25 * z @ (13 * identity - (inputs_bbm_z @ (15 * identity - (inputs_bbm_z @ (7 * identity - inputs_bbm_z)))))
        
        return z