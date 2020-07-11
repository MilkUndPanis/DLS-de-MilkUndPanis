#参数已知的三层网络
from functions import *
class NeuNet:
    def __init__(self):
        # 第一层有一个输入，三个输出，权重分别为0.2,0.4,0.4
        # 偏置分别为0,0.2,-0.2
        self.W0 = np.array([0.2, 0.4, 0.4])
        self.B0 = np.array([0,0.2,-0.2])
        #第一层有三个输入，四个输出，权重分别为0.5,0.8,0.1,0.7
        #                               0.3,0.2,0.6,0.4
        #                               0.2,0.1,0.5,0.7
        #偏置分别为0.3,0.8，-0.6,1.5
        self.W1=np.array([[0.5,0.8,0.1,0.7],[0.3,0.2,0.6,0.4],[0.2,0.1,0.5,0.7]])
        self.B1=np.array([0.3,0.8,-0.6,1.5])
        # 第二层有四个输入，五个输出，权重分别为0.23,0.21,0.29,0.27,0.35
        #                                0.13,0.17,0.22,0.9,0.45
        #                                0.53,0.42,0.46,0.37,0.88
        #                                0.12,0.91,0.46,0.9,0.41
        # 偏置分别为-0.9,2.2,0.8,3.1,1.5
        self.W2 = np.array([[0.23,0.21,0.29,0.27,0.35],[0.13,0.17,0.22,0.9,0.45],
                            [0.53,0.42,0.46,0.37,0.88],[0.12,0.91,0.46,0.9,0.41]])
        self.B2 = np.array([-0.9,2.2,0.8,3.1,1.5])
        # 第三层有五个输入，两个输出，权重分别为0.11，0.57
        #                                0.23,0.79
        #                                0.53,0.57
        #                                0.36,0.82
        #                                0.61，0.69
        # 偏置分别为-3.3,2.1
        self.W3 = np.array([[0.11,0.57],[0.23,0.79],[0.53,0.57],[0.36,0.82],[0.61,0.69]])
        self.B3 = np.array([-3.3,2.1])
        # 第四层有两个输入，10个输出，权重分别为0.11，0.57，0.33,0.39,0.82,0.63,0.34,0.37,0.51,0.40
        #                                0.23,0.79,0.38,0.34,0.35,0.93,0.88,0.67,0.73,0.69
        # 偏置分别为-5.3,-1.4,1.9,1.6,3.5,5.5,2.7,1.4,-8.3,-6.6
        self.W4 = np.array([[0.11,0.57,0.33,0.39,0.82,0.63,0.34,0.37,0.51,0.40],
                            [0.23,0.79,0.38,0.34,0.35,0.93,0.88,0.67,0.73,0.69]])
        self.B4 = np.array([-5.3,1.4,1.9,1.6,3.5,5.5,2.7,1.4,-8.3,-6.6])
        #输入输出值初始化
        self.Output=np.array([])
        self.X0=np.array([])
    def __InputFloor(self,Input):
        self.X0 = np.array(Input)
        self.A0 = np.dot(self.X0, self.W0) + self.B0
        return sigmoid(self.A0)
    def __FirstFloor(self,Input):
        self.X1=self.__InputFloor(Input)
        self.A1=np.dot(self.X1,self.W1)+self.B1
        return sigmoid(self.A1)
    def __FrontFloor(self,Input):
        self.X2=self.__FirstFloor(Input)
        self.A2=np.dot(self.X2,self.W2)+self.B2
        return ReLU(self.A2)
    def __BackFloor(self,Input):
        self.X3=self.__FrontFloor(Input)
        self.A3=np.dot(self.X3,self.W3)+self.B3
        return sigmoid(self.A3)
    def __FinalFloor(self,Input):
        self.X4=self.__BackFloor(Input)
        self.A4=np.dot(self.X4,self.W4)+self.B4
        self.Output=softmax_improved(self.A4)
    def GetInput(self):
        return self.X0
    def WorkingTheNetwork(self,Input):
        self.__FinalFloor(Input)
        return self.Output


