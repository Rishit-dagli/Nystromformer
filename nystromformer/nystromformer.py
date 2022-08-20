import tensorflow as tf

from .nystrom_attention import NystromAttention
from .utils import FeedForward, PreNorm


class Nystromformer(tf.keras.Model):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        attn_values_residual=True,
        attn_values_residual_conv_kernel=33,
        attn_dropout=0.0,
        ff_dropout=0.0,
        **kwargs
    ):
        super(Nystromformer, self).__init__(**kwargs)

        self.depth = depth

        self.net = []
        for _ in range(depth):
            self.net.append(
                PreNorm(
                    NystromAttention(
                        dim=dim,
                        dim_head=dim_head,
                        heads=heads,
                        num_landmarks=num_landmarks,
                        pinv_iterations=pinv_iterations,
                        residual=attn_values_residual,
                        residual_conv_kernel=attn_values_residual_conv_kernel,
                        dropout=attn_dropout,
                    )
                )
            )
            self.net.append(PreNorm(FeedForward(dim=dim, dropout=ff_dropout)))

    def call(self, inputs, mask=None):
        for i in range(self.depth):
            inputs = self.net[2 * i](inputs, mask=mask) + inputs
            inputs = self.net[(2 * i) + 1](inputs) + inputs
        return inputs
