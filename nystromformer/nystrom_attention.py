import math

import tensorflow as tf
from einops import rearrange, reduce

from .utils import MoorePenrosePseudoinverse


class NystromAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.0,
        **kwargs
    ):
        super(NystromAttention, self).__init__(**kwargs)

        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = tf.keras.layers.Dense(
            inner_dim * 3, input_dim=dim, use_bias=False
        )

        self.to_out = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim),
                tf.keras.layers.Dropout(dropout),
            ]
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2

            self.res_conv = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        use_bias=False,
                        groups=heads,
                        kernel_size=(kernel_size, 1),
                        filters=heads * heads,
                        padding="same",
                    ),
                ]
            )

    def call(self, inputs, mask=None, return_attn=False, **kwargs):
        b, n, _, h, m, iters, eps = (
            *inputs.shape,
            self.heads,
            self.num_landmarks,
            self.pinv_iterations,
            self.eps,
        )

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            inputs = tf.pad(
                inputs,
                [[0, 0], [0, 0], [padding, 0], [0, 0], [0, padding], [0, 0]],
                constant_values=0.0,
            )

            if mask is not None:
                mask = tf.pad(mask, [[padding, 0], [0, 0]], constant_values=False)

        q, k, v = tf.split(self.to_qkv(inputs), 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if mask is not None:
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(
                lambda t: t * tf.cast(mask[..., None], dtype=tf.float32), (q, k, v)
            )

        q = q * self.scale

        l = math.ceil(n / m)
        landmark_einops_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_einops_eq, "sum", l=l)
        k_landmarks = reduce(k, landmark_einops_eq, "sum", l=l)

        divisor = l
        if mask is not None:
            mask_landmarks_sum = reduce(
                tf.cast(mask, dtype=tf.float32), "... (n l) -> ... n", "sum", l=l
            )
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        q_landmarks /= divisor
        k_landmarks /= divisor

        einops_eq = "... i d, ... j d -> ... i j"
        sim1 = tf.einsum(einops_eq, q, k_landmarks)
        sim2 = tf.einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = tf.einsum(einops_eq, q_landmarks, k)

        if mask is not None:
            mask_value = -tf.experimental.numpy.finfo(q.dtype).max

            sim1 = tf.where(
                tf.cast(
                    tf.cast(~(mask[..., None]), dtype=tf.float32)
                    * tf.cast(mask_landmarks[..., None, :], dtype=tf.float32),
                    dtype=tf.bool,
                ),
                mask_value,
                sim1,
            )
            sim2 = tf.where(
                tf.cast(
                    tf.cast(~(mask_landmarks[..., None]), dtype=tf.float32)
                    * tf.cast(mask_landmarks[..., None, :], dtype=tf.float32),
                    dtype=tf.bool,
                ),
                mask_value,
                sim2,
            )
            sim3 = tf.where(
                tf.cast(
                    tf.cast(~(mask_landmarks[..., None]), dtype=tf.float32)
                    * tf.cast(mask[..., None, :], dtype=tf.float32),
                    dtype=tf.bool,
                ),
                mask_value,
                sim3,
            )

        attn1, attn2, attn3 = map(
            lambda t: tf.nn.softmax(t, axis=-1), (sim1, sim2, sim3)
        )
        attn2_inv = MoorePenrosePseudoinverse(iteration=iters)(attn2)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        if self.residual:
            out += self.res_conv(v)

        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out
