import random
from itertools import permutations

import numpy as np
import pytest
import tensorflow as tf
from parameterized import parameterized

from .nystrom_attention import NystromAttention


class NystromAttentionTest(tf.test.TestCase):
    def setUp(self):
        super(NystromAttentionTest, self).setUp()

    def generate_params():
        param_list = []
        dim = []
        for _ in range(8):
            dim.append(random.randint(1, 1000))

        param_list = [[a] for a in dim]
        return param_list

    @parameterized.expand(generate_params())
    def test_shape_and_rank(self, dim):
        attention = NystromAttention(
            dim=dim,
            dim_head=64,
            heads=8,
            num_landmarks=256,
            pinv_iterations=6,
            residual=True,
        )
        x = tf.random.uniform((1, 16384, dim))
        mask = tf.ones([1, 16384], dtype=tf.bool)

        y = attention(x, mask=mask)

        self.assertEqual(tf.rank(y), 3)
        self.assertShapeEqual(np.zeros((1, 16384, dim)), y)


if __name__ == "__main__":
    tf.test.main()
