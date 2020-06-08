
import tensorflow as tf
import numpy as np
var=tf.Variable
size=lambda x:x.shape[1:3].as_list()
channal=lambda x:int(x.shape[3])
count=lambda x:int(x.shape[0])
def getFileters(h,w,inc,outc):
    return var(np.random.uniform(size=(h,w,inc,outc)),dtype="float32")

sess=tf.Session()
testtensor=tf.constant(np.random.uniform(size=(5,400,400,5)),dtype="float32")
v=getFileters(5,5,channal(testtensor),5)
t=tf.nn.conv2d(testtensor,filter=v,padding="SAME")
sess.run(tf.global_variables_initializer())

print(sess.run(t))
