import numpy as np
#单位阶跃函数
def step(x):
    return (x>0).astype(np.int)
#sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))
#ReLU函数
def ReLU(x):
    return np.maximum(0,x)
#输出层激活函数之恒等函数
def identify_function(x):
    return x
#输出层激活函数之softmax函数
def softmax(x):
    pre=np.array(x)
    post=[]
    sum=0
    for ak in pre:
        sum+=np.exp(ak)
    for ak in pre:
        post.append(np.exp(ak)/sum)
    return np.array(post)
#改良softmax函数
def softmax_improved(x):
    pre=np.array(x)
    post=[]
    sum=0
    max=np.max(pre)
    for ak in pre:
        sum+=np.exp(ak-max)
    for ak in pre:
        post.append(np.exp(ak-max)/sum)
    return np.array(post)