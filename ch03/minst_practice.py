import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train,), (x_test, t_test) = load_mnist(flatten=True, normalize=False) # minst 데이터를 (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)로 반환
# flatten: 1차원 배열로 정렬 여부, normalize: 0.0~1.0사이의 값으로 치환(정규화) 여부

print(x_train.shape) #(60000, 784) 784 픽셀짜리 그림 데이터가 60,000개 존재 (28*28=784)
print(t_train.shape) #(60000,) 라벨이 60,000존재
print(x_test.shape) #(10000, 784) 784 픽셀짜리 그림 데이터가 10,000개 존재
print(t_test.shape) #(10000,) 라벨이 10,000개 존재