import matplotlib.pyplot as plt
from NetworkExpl import *
def main():
    x=[1,2,3,4]
    y=[2,3,4,5]
    z=[3,4,5,6]
    a=[3,5,6,7]
    arr=np.array([[x,y],[z,a]])
    print(arr)
    print(arr.shape)
    print(arr.dtype)
    X=arr.flatten()
    Y=X>4
    print(Y)
    print(X[Y])
    print((X>4).astype(np.int))
    xaxis=np.arange(-10,10,0.1)
    yaxis1=step(xaxis)
    yaxis2=sigmoid(xaxis)
    yaxis3=ReLU(xaxis)
    yaxis4=softmax(xaxis)
    plt.plot(xaxis,yaxis1,label='step(x)')
    plt.plot(xaxis,yaxis2,label='sigmoid(x)',linestyle='--')
    plt.plot(xaxis,yaxis3,label='ReLU(x)',linestyle='-.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Comparing Step , Sigmoid and ReLU')
    plt.show()
    a1=np.array([[1,3],[2,4]])
    a2=np.array([[3,4],[2,5]])
    a3=np.dot(a1,a2)
    print(a3)
    #神经网络实验
    Input=11
    ProNet=NeuNet()
    Output=ProNet.WorkingTheNetwork(Input)
    print('Following is the input and output of example network:')
    print(Input)
    print(Output)
    print('Finished.')
    OInput=[3,8,10]
    print(softmax(OInput))
    print(softmax_improved(OInput))
    AnotherInput = [0.3, 2.9, 4.0]
    print(softmax(AnotherInput))
    print(softmax_improved(AnotherInput))
main()