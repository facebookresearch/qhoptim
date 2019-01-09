# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

import tensorflow as tf


def build_net(k, use_resource, sparse, seed):
    r = random.Random(seed)
    t = tf.random.normal((k, 1), dtype=tf.float64, seed=0)
    v = tf.Variable(t, use_resource=use_resource)
    v_tensor = v

    if sparse:
        idx = list(range(k))
        r.shuffle(idx)
        v_tensor = tf.gather(v, idx)

    return v, v_tensor


def allclose(x, y, atol, rtol):
    return tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol)


def assert_optimizers_equal(reference_optim_ctor, test_optim_ctor, n=16, k=4, iters=8, tol=1e-7):
    net_params = [(k, False, False, 0), (k, False, True, 0), (k, True, False, 0), (k, True, True, 0)]

    config = tf.ConfigProto(device_count={"GPU": 0})

    for np in net_params:
        v1, v1_tensor = build_net(*np)
        v2, v2_tensor = build_net(*np)

        reference_optim = reference_optim_ctor()
        test_optim = test_optim_ctor()

        coeffs = tf.random.normal((k, 1), dtype=tf.float64, seed=1)

        x = tf.placeholder(tf.float64, [n, k])
        y = tf.placeholder(tf.float64, [n, 1])

        reference_output = tf.matmul(x, v1_tensor)
        test_output = tf.matmul(x, v2_tensor)

        reference_loss = tf.math.reduce_mean((reference_output - y) ** 2)
        test_loss = tf.math.reduce_mean((test_output - y) ** 2)

        reference_min = reference_optim.minimize(reference_loss, var_list=[v1])
        test_min = test_optim.minimize(test_loss, var_list=[v2])

        check = allclose(v2, v1, atol=tol, rtol=tol)

        with tf.Session(config=config) as sess:
            sess.run(v1.initializer)
            sess.run(v2.initializer)

            sess.run(tf.variables_initializer(reference_optim.variables()))
            sess.run(tf.variables_initializer(test_optim.variables()))

            for it in range(iters):
                x_data = sess.run(tf.random.normal((n, k), dtype=tf.float64, seed=2 + it))
                y_data = sess.run(
                    tf.matmul(x_data, coeffs) + tf.random.normal((n, 1), dtype=tf.float64, seed=(2 ** 30) + it) + 5.0
                )

                feed_dict = {x: x_data, y: y_data}
                sess.run([reference_min, test_min], feed_dict=feed_dict)

                assert sess.run(check)
