# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

class Network:
    def __init__(self, Nn=100, Hln=5):
        self.activations={}
        # FCNN의 한 계층당 노드 수dd
        self.node_num = Nn
        # 은닉층 수
        self.hidden_layer_num = Hln
        # 활성화값 {계층0:활성화값배열, 계층1:활성화값배열,....}
        self.activations = {}

    def weight_init_activation_exp(self, En=100, stddev = 1):

        # 100 원소가 한 엔트리를 구성하는 En 개 엔트리의 데이터셋
        x = np.random.randn(En, self.node_num)

        for i in range (self.hidden_layer_num):
            if i !=0 :
                x = self.activations[i-1] # 최초를 빼고는 이전 계층의 활성화값이 입력값임

            # 계층간 가중치매트릭스는 임의로 생성한다
            w = np.random.randn(self.node_num, self.node_num) * stddev
            a = np.dot(x,w)
            z = sigmoid(a)
            self.activations[i] = z
#            print (self.activations[i])
#            print (self.activations[i].shape)

    def draw_histogram(self):
        for i, a in self.activations.items():
            plt.subplot(1,len(self.activations), i+1)
            plt.title (str(i+1) + '-layer')
            plt.hist (a.flatten(), 30, range= (0,1))
        plt.show()

def run_exp():
    net = Network()
    net.weight_init_activation_exp(1000, 1/np.sqrt(100))
    net.draw_histogram()


def main():
    run_exp()

if __name__ == "__main__":
    main()


