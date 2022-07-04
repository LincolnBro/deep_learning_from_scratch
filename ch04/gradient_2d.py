import numpy as np

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def _numerical_gradient_no_batch(f, x): # 한 번에 한 좌표만 계산하는 경우, x는 ndim=1인 좌표 벡터 [x,y,z], grad에는 해당좌표의 편미분 값이 들어감 [fx, fy, fz]

    h=1e-4 #0.0001
    grad=np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx]=tmp_val+h
        fxh1 = f(x)
        # f(x-h) 계산
        x[idx]=tmp_val-h
        fxh2 = f(x)

        grad[idx]=(fxh1 - fxh2) / (2*h)
        x[idx]=tmp_val

    return grad

def numerical_gradient(f, X):
    if X.ndim == 1: # 한 번에 한 좌표만 계산하는 경우
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X) # 한 번의 배치 단위 좌표를 계산하는 경우, X=[[x],[y],[z]]
        
        for idx, x in enumerate(X): # X 각 index의 값을 index와 같이 반환, X를 ndim=1짜리 벡터들로 쪼개서 계산
            grad[idx] = _numerical_gradient_no_batch(f, x) # grad=[[fx], [fy], [fz]]
        
        return grad