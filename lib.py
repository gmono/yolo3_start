import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
def unpool(input):
    #最大池化
    pool1 = tf.nn.max_pool(input,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    #最大反池化
    grad = gen_nn_ops.max_pool_grad(input, #池化前的tensor，即max pool的输入
                                    input, #池化后的tensor，即max pool 的输出
                                    input, #需要进行反池化操作的tensor，可以是任意shape和pool1一样的tensor
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
    return  grad


def max_unpool_2x2(x):
    shape=x.shape.as_list()
    inference = tf.image.resize_nearest_neighbor(x, [shape[1]*2, shape[2]*2])
    return inference
