import random
from itertools import permutations

import numpy as np
import pytest
import tensorflow as tf
from parameterized import parameterized

from .nystromformer import Nystromformer


class NystromformerTest(tf.test.TestCase):
    def setUp(self):
        super(NystromformerTest, self).setUp()

    def generate_params():
        param_list = []
        dim = []
        for _ in range(8):
            dim.append(random.randint(1, 1000))

        param_list = [[a] for a in dim]
        return param_list

    @parameterized.expand(generate_params())
    def test_shape_and_rank(self, dim):
        model = Nystromformer(
            dim=dim, dim_head=64, heads=8, depth=6, num_landmarks=256, pinv_iterations=6
        )

        x = tf.random.uniform((1, 16384, dim))
        mask = tf.ones([1, 16384], dtype=tf.bool)

        y = model(x, mask=mask)

        self.assertEqual(tf.rank(y), 3)
        self.assertShapeEqual(np.zeros((1, 16384, dim)), y)


if __name__ == "__main__":
    tf.test.main()
