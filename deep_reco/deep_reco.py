import os.path
import os

import tensorflow as tf
from tensorflow.python.framework import ops
from Data.input_helper import get_input_output_pairs
import numpy as np
import lme_custom_ops

num_training = 100
learning_rate = 0.001

input_shape = (370, 256)
output_shape = (256, 256)


@ops.RegisterGradient( "ParallelProjection2D" )
def _project_grad( op, grad ):
    reco = lme_custom_ops.parallel_backprojection2d(
            sinogram                      = grad,
            sinogram_shape              = op.get_attr("projection_shape"),
            volume_shape                = op.get_attr( "volume_shape" ),
            volume_origin               = op.get_attr( "volume_origin" ),
            detector_origin             = op.get_attr( "detector_origin" ),
            volume_spacing              = op.get_attr( "volume_spacing" ),
            detector_spacing            = op.get_attr( "detector_spacing" ),
            ray_vectors                 = op.get_attr( "ray_vectors" ),
        )
    return [ reco ]

def run_training():
    '''
    :return: tf.Graph object
    '''
    BP_graph = tf.Graph()
    adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
    with BP_graph.as_default():
        # variables and placeholders
        sinogram = tf.placeholder(dtype=tf.float32, shape=input_shape, name='sinogramm_in')
        K = tf.Variable(tf.truncated_normal(input_shape), name='Kernel_to_learn', trainable=True)
        target = tf.placeholder(dtype=tf.float32, shape=output_shape, name='target_image')

        # layers
        fourier = tf.spectral.fft2d(sinogram)
        multiply = tf.multiply(fourier, K, name='trainable_layer')
        inf_fourier = tf.spectral.ifft2d(multiply)
        out = lme_custom_ops.parallel_projection2d(updated_reco,
                                                  volume_shape,
                                                  sinogram_shape,
                                                  volume_origin,
                                                  detector_origin,
                                                  volume_spacing,
                                                  detector_spacing,
                                                  rays_tensor
                                                  )
        # loss and optimizer
        loss = tf.reduce_sum(tf.squared_difference(out, target))
        train = adam.optimize(loss)


    with tf.Session(graph=BP_graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        sino, truth = get_input_output_pairs()
        loss = sess.run(train, feed_dict={sinogram: sino, target: truth})
        print('ergebnis: ')
        print(out)


def test_run():
    custom_Graph = tf.Graph()

    with custom_Graph.as_default():
        a = tf.placeholder(dtype=tf.float32, name='input')
        b = tf.Variable(5, dtype=tf.float32)
        c = tf.multiply(a, b)

    with tf.Session(graph=custom_Graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        res = sess.run(c, feed_dict={a: np.array(4)})
        print(res)


if __name__ == '__main__':
    run_training()
    print('done')
    os._exit(0)
