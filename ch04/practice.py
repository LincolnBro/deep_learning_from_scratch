import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 신경망 생성
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼 파라미터
iters_num = 100000  # 반복 횟수 설정
train_size = x_train.shape[0]
print(train_size)
batch_size = 100   # 미니배치 크기
learning_rate = 0.1 # 학습률 설정

train_loss_list = []
train_acc_list = []
test_acc_list = []