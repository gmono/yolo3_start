# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf

# %%
import numpy as np

# %%
from typing import List, Dict


def debug(func):
    def r(*args, **kwargs):
        print(locals())
        return func(*args, **kwargs)

    return r


# %%
# @emit 体系
# @emit(pathname,"a","b","c") __globalConfig字典 使用参数顶替策略 
# pathname采用jsonpath，构建层次对象结构，以扁平化方式写入文件，使用json or yaml or toml

# %%
var = tf.Variable
size = lambda x: x.shape[1:3].as_list()
channel = lambda x: int(x.shape[3])
count = lambda x: int(x.shape[0])


# %%
@debug
def getFileters(h, w, inc, outc):
    return var(np.random.uniform(low=-1, high=1, size=(h, w, inc, outc)), dtype="float32")


# %%
def calculateFilterSize(output_size, input_size):
    c = lambda o, i: i - o + 1
    return [c(output_size[0], input_size[0]), c(output_size[1], input_size[1])]


actstb={
    "sigmoid":tf.sigmoid,
    "relu":tf.nn.relu,
    "tanh":tf.tanh,
    "cos":tf.cos,
    "softplus":tf.nn.softplus,
    "leaky":tf.nn.leaky_relu,
    "linear":lambda x:x
}
def activate(name):
    assert name in actstb,f"activation function is not supported now," \
                          f"you can use the following activation functions:{actstb.keys()}"
    return actstb[name]
# %%
## mapping functions series starts here
# map hwcs -> conv
# here the "hwcs" respersent the paramters of this function
# temporarily use sigmoid function to calculate the activation value
# the shape and value of the paramters in any layer  will be determined when called for the first time
def conv(h, w, c, stride=1, padding="SAME",activateFunc=activate("leaky")):
    """
    h w c means the size of the filter
    this is not equal to the concept of "triple" mentioned later
    """
    f=None
    norm=batchnorm()
    def run(tensor):
        nonlocal f
        if f is None:
            f = getFileters(h, w, channel(tensor), c)
        x = tf.nn.conv2d(input=tensor, filter=f, strides=[1, stride, stride, 1], padding="SAME")
        x = activateFunc(x)
        # use batch normalization function
        x=norm(x)
        return x

    return run

def batchnorm():
    """
    create a batchnorm layer
    :return: a function represent a batchnorm layer
    """
    scale=None
    shift=None
    epsilon = 0.001
    def run(tensor):
        nonlocal  scale
        nonlocal  shift
        nonlocal  epsilon
        mean, vars = tf.nn.moments(x, axes=[0])
        # size will determined by  the shape of sample  such as (416,416,3)
        # and then the shape of the scale and shift would be (1,416,416,3)
        # which represented
        if scale is None:
            size = tensor[0].shape
            scale = tf.Variable(tf.ones([size]))
            shift = tf.Variable(tf.zeros([size]))
        return tf.nn.batch_normalization(tensor,mean,vars,shift,scale,epsilon)
    return run

# %%
# map [hwcs]->[conv]
def convs(listofparamter):
    convs = list(map(lambda triple: conv(*triple), listofparamter))
    return convs


# nested function
# conv(hwcs)(tensor)()
# repeat conv can create 


# %%
# map [func]->func(seq) including [conv]->func(seq)
def combine(layers: list):
    """
    convs:list of function , function here is generated by conv function
    standard:the output of previous one should be 
    """

    def run(tensor):
        now = tensor
        for func in layers:
            now = func(now)
        return now

    return run


# %%
# map func->func with residual connection
def residualize(fun):
    """
    this function would pack the function in paramters(fun) by add a cross-layer connections
        from start position to end position
    seq there is defined as a function that is just like "func(input)->output"
    the input and output of seq should 
    """

    def run(tensor):
        x = fun(tensor)
        return tensor + x

    return run















# pa
# %%
# new starting point
# generate functions serise starts here
# generate functions can generate a list representing the paramters of a type of layer
# for example ,the paramters of  conv layer can be represent as a hexagram such as [hwcspa]
# usually these functions would receive their own paramters,as well as a list

# all these functions will  have a paramter named otherParamterList or otherParamter
# in which there is template of the paramter of layer
# question: how to represent a paramter table of a layer?
# can we use a list and a dict to do that?
# can we replace
def get_ShapePretected_Paramters(numberOfConvs, outputShape, otherParamterList=[(1, 1, 0), (3, 3, 0)]):
    """
    generation function
    this function would generate a list of triples that represent a series of conv layer ,this serise of conv layer
    would not change the shape of inputtensor and its order follows staggered distribution of size
    :param outputShape represent
    """
    assert numberOfConvs % 2 == 0, "the number of layer must be even"
    # temporarily adopt the strategy of  "single shrinkage middle layer" 
    # randomly generate a list of triples
    c = outputShape[3]
    # this function's function  have been changed to calculate the size of kernel with different number of channels
    # even corresponds to half
    # odd corresponds to all
    lst = [
        c // 2,
        c
    ]

    def randomTriple(idx):
        # select kernel's size : h and w
        ptri = list(otherParamterList[idx % len(otherParamterList)])
        # select channel
        ptri[2] = lst[idx % 2]
        return ptri

    ret = list(map(lambda x: randomTriple(x), range(numberOfConvs)))
    return ret

def get_Shrink_ConvTriple(inputtensor,h=3,w=3):
    return [h, w, channel(inputtensor) * 2, 2]


# %%
# utility function area
def yolo_resblock():
    def run(tensor):
        with tf.name_scope("resBlock"):
            lst = get_ShapePretected_Paramters(numberOfConvs=2, outputShape=tensor.shape.as_list())
            layers=convs(lst)
            seq=combine(layers)
            resb = residualize(seq)
            return resb(tensor)
    return run

def yolo_shrinkLayer():
    def run(tensor):
        #par is a quad
        with tf.name_scope("shrinkLayer"):
            par=get_Shrink_ConvTriple(inputtensor=tensor)
            layer=conv(*par)
            return layer(tensor)
    return run


def yolo_proShapeConvs(outShape=None):
    """
    usually use by output layers in yolo3
    :return: the function representing a function that would map input to other with the same shape
    """
    def run(tensor):
        if outShape is not None:
            s=outShape
        else:
            s=tensor.shape.as_list()
        with tf.name_scope("proShapeConvs"):
            lst = get_ShapePretected_Paramters(numberOfConvs=2, outputShape=s)
            layers=convs(lst)
            seq=combine(layers)
            return seq(tensor)
    return run

def yolo_doubleOutputLayer():
    """
    usually use in the output layer in yolo3
    :return: (middleoutput,output) the function that represent a function that would map input to other with the same shape and double output
            ,which means it would output a result of all layer in seq as well as a middle output
            ,usually comes from the shrinkage channel layer
    """
    def run(tensor):
        with tf.name_scope("doubleOutputLayer"):
            lst = get_ShapePretected_Paramters(numberOfConvs=2, outputShape=tensor.shape.as_list())
            layers = convs(lst)
            x=layers[0](tensor)
            y=layers[1](x)
            return x,y
    return run

def yolo_headLayer():
    def run(tensor):
        with tf.name_scope("headLayer"):
            # keep size unchanged,channel change to 32
            # record the input data on head
            tf.summary.image("inputImages",tensor)
            layer=conv(3,3,c=32)
            return layer(tensor)
    return run

def yolo_outputLayer():
    def run(tensor):
        # 1 1 1 255
        with tf.name_scope("outputLayer"):
            layer=conv(1,1,255,activateFunc=activate("linear"))
            # the activation function will calculate in yolo_loss layer with loss
            return layer(tensor)
    return run

def yolo_loss():
    def run(output,label):
        # will record the loss value by tf.summary.scalar there
        # there will be calculations for the loss and the activation value of output
        pass
    return run
# %%
sess = tf.Session()

# %%
from matplotlib import  pyplot as plt
img=plt.imread("壁纸.jpg")#type:np.ndarray
img=img.astype("float32")/255


testtensor =tf.constant(img,dtype="float32")
testtensor=tf.expand_dims(testtensor,axis=0)
print(testtensor.shape)
testtensor=tf.image.resize_images(testtensor,(416,416))
# %%
lst = get_ShapePretected_Paramters(2, outputShape=testtensor.shape.as_list())


# %%
inputsize = 416
inputch = 3

# %%
# cls区域
from lib import  max_unpool_2x2

class Yolo3():
    # 特征金字塔 从大尺寸 少通道 -> 小尺寸 大通道
    # features here is used by subgraph
    # 从末尾取两个通过子图处理后得到一个中间输出，再添加到末尾，如此反复直到len小于2 （待翻译

    # start
    # 主网络有3个特征输出和一个最终输出
    # 子网络有一个特征输出和一个最终输出
    # 最后一个自网络只有一个最终输出，其特征输出丢弃，但计算时依然保存其特征输出

    def __init__(self):
        super(Yolo3).__init__()
        self.mid_features = []  # type:List[tf.Tensor]
        self.outputs=[] #type:List[tf.Tensor]
    @staticmethod
    def stackResblock(count: int, x: tf.Tensor):
        with tf.name_scope("stackResblocks"):
            for i in range(count):
                x = yolo_resblock()(x)
        return x

    def outputProcess(self, x:tf.Tensor, featureShape=None):
        """
        this function would do the output process and insert the middle feature into the end of mid_features
        ,and then return the output
        :param x: input tensor
        :return: the output
        """
        with tf.name_scope("outputProcess"):
            #output process
            x=yolo_proShapeConvs(featureShape)(x)
            x=yolo_proShapeConvs()(x)
            # subgraph feature output
            mdfeature,outfeature=yolo_doubleOutputLayer()(x) #type:tf.Tensor
            # middle feature is used for subgraph
            # assert mdfeature.shape.as_list()[1:]==[13,13,512]
            self.mid_features.append(mdfeature)
            # output feature is used for output layer
            output=yolo_outputLayer()(outfeature)
            # the output above would be used for detection
        return output

    def mainNet(self,tensor:tf.Tensor):
        """
        perform the feature extraction process
        :param tensor: input tensor
        :return:
        """
        with tf.name_scope("mainNet"):
            x=yolo_headLayer()(tensor)
            x=yolo_shrinkLayer()(x)
            # res1
            x = yolo_resblock()(x)
            # shrink
            x=yolo_shrinkLayer()(x)
            # res 2 to 3
            x= self.stackResblock(2, x)
            # shrink
            x=yolo_shrinkLayer()(x)
            # res 4 to 11 count:8
            x= self.stackResblock(8, x)
            # output 52 52 256
            assert x.shape.as_list()[1:]==[52,52,256],f"feature shape is wrong,the shape is {x.shape.as_list()}"
            self.mid_features.append(x)
            # shrink
            x=yolo_shrinkLayer()(x)
            # res 12 to 18
            x= self.stackResblock(7, x)
            assert x.shape.as_list()[1:]==[26,26,512],f"feature shape is wrong,the shape is {x.shape.as_list()}"
            self.mid_features.append(x)
            # shrink
            x=yolo_shrinkLayer()(x) # here the feature's shape has been changed to [13,13,1024]
            # res 19 to 22
            x= self.stackResblock(4, x)
            #############
            out= self.outputProcess(x)
            return out


    def subNet(self,amplificationInput,directInput):
        # two subnet here is Identical
        # channel shrink -> upsampling ->concat -> output process
        with tf.name_scope("subNet"):
            shrinkChannel=conv(1,1,channel(amplificationInput)//2)
            #start
            x=shrinkChannel(amplificationInput)
            #upsample
            x=max_unpool_2x2(x)
            print(x.shape.as_list())
            assert size(x)==size(directInput),f"feature shape is wrong,the shape is {x.shape.as_list()} and {directInput.shape.as_list()}"
            # concat
            concatx=tf.concat([x,directInput],axis=3) #type:tf.Tensor
            assert channel(concatx)==channel(directInput)/2*3,f"channel is wrong,the shape is {concatx.shape.as_list()} and {directInput.shape.as_list()}"

            # enter the output processing
            startshape=concatx.shape.as_list()
            startshape[3]=startshape[3]//3*2
            output= self.outputProcess(concatx, featureShape=startshape)
            # here ,the new midlle-feature was inserted into the end of  mid_features array
            return output

    def initialize(self):
        self.mid_features=[]
        self.outputs=[]
    def run(self,input):
        mainout=self.mainNet(input)
        self.outputs.append(mainout)
        with tf.name_scope("SubNets"):
            for i in range(2):
                if len(self.mid_features)<2:break
                dir,amp=self.mid_features[-2:]
                del self.mid_features[-2:]
                self.outputs.append(self.subNet(amplificationInput=amp,directInput=dir))
            return self.outputs

    def printState(self):
        print("inner states:")
        print("midfeatures:",[i.shape.as_list() for i in self.mid_features])
        print("outputs:",[i.shape.as_list() for i in self.outputs])

## 子网络阶段 concat any two adjacent features in sequence
#扩增+融合

# %%
yolo=Yolo3()
x=yolo.run(testtensor)
yolo.printState()
# x=yolo_headLayer()(testtensor)

# %%
initop = tf.global_variables_initializer()
sess.run(initop)

# %%

print(x)
sess.run(x)

summray=sess.run(tf.summary.merge_all())



# %%
# here is the tensorboard section
from tensorflow.summary import FileWriter

writer = FileWriter("./logs", sess.graph)
writer.add_summary(summary=summray)
writer.close()
import os
try:
    oracle_vars = dict((a, b) for a, b in os.environ.items() if a.find('IPYTHONENABLE') >= 0)
    if len(oracle_vars) >0:
        print(oracle_vars)
        print(os.environ['IPYTHONENABLE'])
        print(os.environ['IDE_PROJECT_ROOTS'])
        print(os.environ['PYTHONDONTWRITEBYTECODE'])
except ZeroDivisionError as e:
    print(e.message)
    print("????")
