#!/usr/bin/env python
# coding: utf-8

# # Repeat 노드

# In[9]:


import numpy as np

# # 순전파

# D = 8
# N = 7
# x = np.random.rand(1,D)  # (1,8)
# # x = np.random.rand(D,).reshape(1,-1)  # (1,8)
# print(x,x.shape)
# print('-'*70)

# y = np.repeat(x,N,axis=0)  # 수직(행) 방향, axis=0
# print(y,y.shape)   # (7, 8)


# # In[15]:


# # 역전파 : sum
# dy = np.random.rand(N,D)
# print(dy,dy.shape)  # (7, 8)
# print('-'*70)
# dx = np.sum(dy,axis=0,keepdims=True)  # 수직방향 합, keepdims=True이면 2차원, False이면 1차원
# print(dx,dx.shape)  # (1,8)


# # In[18]:


# a = np.array([[1,2,3,4]])
# np.sum(a, keepdims=True) # 2차원 유지


# # ### Sum 노드

# # In[22]:


# # 순전파

# D,N = 8,7
# x = np.random.rand(N,D) 
# print(x,x.shape)  # (7,8)
# print('-'*70)

# y = np.sum(x,axis=0,keepdims=True)  # 수직방향 합, keepdims=True이면 2차원, False이면 1차원
# print(y,y.shape)


# # In[23]:


# # 역전파

# dy = np.random.rand(1,D)  # (1,8)
# print(dy,dy.shape)
# print('-'*70)

# dx = np.repeat(dy,N,axis=0)  # 수직(행) 방향, axis=0
# print(dx,dx.shape)   # (7, 8)


# ### MatMul 노드

# In[24]:


class MatMul:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
        
    def forward(self,x):
        W, = self.params
        out = np.dot(x,W)
        self.x = x
        return out
        
    def backward(self,dout):
        W, = self.params
        dx = np.dot(dout,W.T)
        dw = np.dot(self.x.T,dout)
        self.grads[0][...] = dw  # 깊은복사
        return dx


# In[27]:


# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# print(hex(id(a)),hex(id(b)))
# a = b    # 얕은 복사
# print(a)
# print(hex(id(a)),hex(id(b)))
# id(a) == id(b)


# In[28]:


# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# a[...] = b  # 깊은 복사
# print(a)
# print(hex(id(a)),hex(id(b)))
# id(a) == id(b)


# In[31]:


# # np.zeros_like
# a = np.arange(12).reshape(3,4)
# b = np.zeros_like(a)
# b


# sigmoid 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


### 시그모이드 계층

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [],[]
        self.out = None
        
    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self,dout):
        dx = dout*self.out*(1 - self.out)  # 공식 도출은 참고서적 참조
        return dx        



### ReLU 계층  

class ReLU:
    def __init__(self):
        self.params, self.grads = [], []
        self.mask = None
        self.out = None

    def forward(self, x):
        self.mask = (x <= 0)  # x가 0 이하일 경우 0으로 변경
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0  #  x가 0 이하일 경우 0으로 변경
        dx = dout 
        return dx


# ### Affine 계층 : MatMul 노드에 bias를 더한 계층,  X*W + b

# In[33]:


class Affine:
    def __init__(self,W,b):
        self.params = [W,b]
        self.grads = [np.zeros_like(W),np.zeros_like(b)]
        self.x = None
        
    def forward(self,x):
        W , b = self.params
        out = np.dot(x,W) + b
        self.x = x
        return out
        
    def backward(self,dout):
        W, b = self.params
        dx = np.dot(dout,W.T)
        dw = np.dot(self.x.T,dout)
        db = np.sum(dout,axis=0)
        
        self.grads[0][...] = dw  # 깊은복사
        self.grads[1][...] = db  # 깊은복사
        
        return dx    


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x    

# Softmax 계층
class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx

# ## Softmax with Loss 계층

# In[35]:


class SoftmaxWithLoss:
    def __init__(self):
        self.params,self.grads=[],[]
        self.y = None  # softmax의 출력값
        self.t = None  # 정답 레이블
        
    def softmax(self,x):
        if x.ndim == 2:
            x = x - x.max(axis=1, keepdims=True)  # nan출력을 방지
            x = np.exp(x)
            x /= x.sum(axis=1,keepdims=True)
        elif x.ndim == 1:
            x = x - np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))
        return x  
    
    # https://smile2x.tistory.com/entry/softmax-crossentropy-%EC%97%90-%EB%8C%80%ED%95%98%EC%97%AC 
    def cross_entropy_error(self,y, t):  
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]

        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  # 1e-7은 log(0)으로 무한대가 나오는걸 방지
          
    
    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
       
        loss = self.cross_entropy_error(self.y,self.t)
        return loss
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        
        # dx = (self.y - self.t)/batch_size  # 순수 Softmax계층 일경우
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        
        return dx       
        


# In[36]:


# # softmax 구현시에  지수값이 크면 오버플로발생으로 nan이 나오는 것을 방지하기 위해 입력 값의 촤대값을 빼주어 사용한다
# a = np.array([1010,1000,990])
# print(np.exp(a))    # [inf inf inf]  , 무한대 값, 오버플로우 발생
# x = np.exp(a)/np.sum(np.exp(a))
# print(x)  # [nan nan nan]

# c = np.max(a)
# print(a - c)
# x2 = np.exp(a - c)/np.sum(np.exp(a - c))
# print(x2)  # [9.99954600e-01 4.53978686e-05 2.06106005e-09]


# ### 가중치 갱신

# In[38]:


# 확률적 경사하강법(Stochastic Gradient Descent)
class SGD :
    def __init__(self,lr=0.01):
        self.lr = lr
        
    def update(self,params,grads):
        for i in range(len(params)):
            params[i] -= self.lr*grads[i]    


# In[ ]:


class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
            

import time
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

def remove_duplicate(params, grads):
    '''
    매개변수의 중복 제거 함수
    매개변수 배열 중 중복되는 가중치를 하나로 모아
    그 가중치에 대응하는 기울기를 더한다.
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시  : lSTM에서 사용
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # 뒤섞기
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # 기울기 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                
                params, grads = remove_duplicate(model.params, model.grads)  # 공유된 가중치를 하나로 모음
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 손실 %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('반복 (x' + str(self.eval_interval) + ')')
        plt.ylabel('손실')
        plt.show()      



# Embedding 계층
class Embedding :
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
    
    # 순전파
    def forward(self,idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    # 역전파
    def backward(self, dout):  # 중복 인덱스가 있어도 올바르게 처리, 속도가 빠름
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)  
        return None       

    
# EmbeddingDot 계층

class EmbeddingDot:
    def __init__(self,W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
        
    def forward(self,h,idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W*h,axis=1)   # 1차원 출력
        self.cache = (h, target_W)
        return out
    
    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0],1) # 2차원으로 변환
        
        dtarget_W = dout*h  # sum <--> repeat, 브로드캐스트
        self.embed.backward(dtarget_W)
        
        dh = dout*target_W  # 브로드캐스트
        return dh
    


# UnigramSampler

import collections
class UnigramSampler:    
    # 생성자 : corpus를 사용하여 단어의 0.75제곱 처리한 확률 분포를 구함
    def __init__(self, corpus, power, sample_size): # power= 0.75, sample_size = 2
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        # corpus 내의 단어별 발생횟수를 구함    
        counts = collections.Counter()  
        for word_id in corpus:   # corpus: [0 1 2 3 4 1 5 6], 
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size  # 7

        self.word_p = np.zeros(vocab_size)  # (7,)
        for i in range(vocab_size):  # 7 
            self.word_p[i] = counts[i]  # [1, 2, 1, 1, 1, 1, 1] ,단어 발생 횟수

        self.word_p = np.power(self.word_p, power) # 0.75제곱
        self.word_p /= np.sum(self.word_p)  # 전체의 합으로 나누어 확률을 구함


    def get_negative_sample(self, target):   # target = np.array([1, 3, 0]), (3,)
        batch_size = target.shape[0]  # 3
        
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)  # (3,2)

        for i in range(batch_size):  # 3회
            p = self.word_p.copy()
            target_idx = target[i]  # 1,3,0
            p[target_idx] = 0  # p[1]=0,p[3]=0,p[0]=0 ,부정 단어로 target이 선택되지 않도록 확률 값을 0으로 설정 
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
            
        return negative_sample
    

# SigmoidWithLoss 클래스 사용 

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid의 출력
        self.t = None  # 정답 데이터

    def cross_entropy_error(self,y, t):   # softmax 와 동일
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]

        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
        
    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))   # sigmoid , 예측값

        self.loss = self.cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx
    

# NegativeSamplingLoss 클래스

class NegativeSamplingLoss:
    def __init__(self,W,corpus,power=0.75,sample_size=5): #  sample_size : 부정 단어 샘플링 수 (2개)
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus,power,sample_size)
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)] # 긍정 1개 + 부정 2개
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]   # 긍정 1개 + 부정 2개
        
        self.params, self.grads = [],[]
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def forward(self,h,target) : # target은 긍정단어의 index
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target) # 부정 단어 샘플
        
        # 긍정단어 순전파
        score = self.embed_dot_layers[0].forward(h,target)  # target-->index
        correct_label = np.ones(batch_size, dtype=np.int32) # 값이 모두 1: 긍정
        loss = self.loss_layers[0].forward(score,correct_label)        
        
        # 부정단어 순전파
        negative_label = np.zeros(batch_size,dtype=np.int32) # 값이 모두 0 : 부정 
        
        for i in range(self.sample_size):
            negative_target = negative_sample[:,i]
            score = self.embed_dot_layers[i + 1].forward(h,negative_target)
            loss += self.loss_layers[i + 1].forward(score,negative_label) # loss의 누적 합
            
        return loss
     
    def backward(self,dout=1)  : # 입력값을 각 계층의 backward만 호출하여 전달
        dh = 0
        for l0,l1 in zip(self.loss_layers, self.embed_dot_layers):  # 역전파이므로  los_layer가 먼저 호출된다
            dscore = l0.backward(dout)   # SigmoidWithLoss 계층
            dh += l1.backward(dscore)   # EmbeddingDot 계층   
        return dh

    
# 기울기 클리핑 함수    
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:  # L2 norm 구하기 , 제곱의 합의 제곱근
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    # print('rate:',rate)
    if rate < 1:     # total_norm 이 한계값(max_norm) 보다 클경우
        for grad in grads:
            grad *= rate
            

            
# TimeEmbedding 클래스
class TimeEmbedding:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W
    
    def forward(self,xs):  # N : batch size, T : sequence length
        N,T = xs.shape
        V,D = self.W.shape  # V : Hidden size, 사용되지 않음  , D : RNN input size
        
        out = np.empty((N,T,D),dtype='f')   # W:(V,D) -> xs:(N,T) ==> (N,T,D)
        self.layers = []
        
        for t in range(T):
            layer = Embedding(self.W)
            out[:,t,:] = layer.forward(xs[:,t])
            self.layers.append(layer)
            
        return out  # RNN 계층으로 입력, (N,T,D) 3차원  
    
    def backward(self, dout) :
        N, T, D = dout.shape
        
        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:,t,:])
            grad += layer.grads[0]
        
        self.grads[0][...] = grad
        return None              

    
# TimeAffine 계층
class TimeAffine:
    def __init__(self,W,b):
        self.params = [W,b]
        self.grads = [np.zeros_like(W),np.zeros_like(b)]
        self.x = None
        
    def forward(self,x):
        N,T,D = x.shape   # N : batch size, T : sequence length , D : Affine input size,RNN의 출력사이즈
        W,b = self.params
        
        rx = x.reshape(N*T,-1)  # 2차원으로 변경하여 내적 곱
        out = np.dot(rx,W) + b
        self.x = x
        
        return out.reshape(N,T,-1) # 3차원으로 변경하여 출력

    def backward(self, dout):  # dout : 3차원
        x = self.x
        N, T, D = x.shape
        W, b = self.params
        
        dout = dout.reshape(N*T, -1)  #  2차원으로 변경
        rx = x.reshape(N*T,-1)        #  2차원으로 변경
        
        db = np.sum(dout,axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout,W.T)
        dx = dx.reshape(*x.shape)  # 튜플 가변인수,(N,T,D)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        
        return dx        
    


# TimeSoftmaxWithLoss 계층
class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def softmax(self,x):
        if x.ndim == 2:
            x = x - x.max(axis=1, keepdims=True)
            x = np.exp(x)
            x /= x.sum(axis=1, keepdims=True)
        elif x.ndim == 1:
            x = x - np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))

        return x        

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 정답 레이블이 원핫 벡터인 경우
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)  # label 이 -1이 아닌 값만 True 가  되는 2차원 배열을 생성

        # 배치용과 시계열용을 정리(reshape)
        xs = xs.reshape(N * T, V)   # 2차원으로
        ts = ts.reshape(N * T)      # 1차원으로
        mask = mask.reshape(N * T)  # 1차원으로 

        ys = self.softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask                # ignore_label(-1)에 해당하는 데이터는 손실을 0으로 설정
        loss = -np.sum(ls)        # T개의 loss를 합한다
        loss /= mask.sum()        # 평균

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys        # softmax의 출력 값과
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_label(-1)에 해당하는 데이터는 기울기를 0으로 설정, mask를 2차원으로 변경하여 곱해줌

        dx = dx.reshape((N, T, V)) # 다시 3차원으로 변경

        return dx
    
    
# RNNLM의 Trainer 클래스
import time

class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]  # 배치에서 각 샘플을 읽기 시작하는 위치

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35,
            max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 기울기를 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 공유된 가중치를 하나로 모음
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 퍼플렉서티 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 퍼플렉서티 %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('반복 (x' + str(self.eval_interval) + ')')
        plt.ylabel('퍼플렉서티')
        plt.show()

def remove_duplicate(params, grads):
    '''
    매개변수 배열 중 중복되는 가중치를 하나로 모아
    그 가중치에 대응하는 기울기를 더한다.
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads
            

# clip_grads()함수 : 기울기 클리핑
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:  # L2 norm 구하기 , 제곱의 합의 제곱근
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    # print('rate:',rate)
    if rate < 1:     # total_norm 이 한계값(max_norm) 보다 클경우
        for grad in grads:
            grad *= rate    


class LSTM:
    def __init__(self,Wx,Wh,b):
        '''
        Parameters
        ----------
        Wx: 입력 x에 대한 가중치 매개변수(4개분의 가중치가 담겨 있음)
        Wh: 은닉 상태 h에 대한 가중치 매개변수(4개분의 가중치가 담겨 있음)
        b: 편향（4개분의 편향이 담겨 있음）
        '''
        self.params = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
        
    def forward(self,x,h_prev,c_prev) :
        Wx, Wh, b = self.params           # Wx,Wh : (D,4*H)로 생성하여 입력됨
        N, H = h_prev.shape
        
        A = np.dot(x,Wx) + np.dot(h_prev,Wh) + b
        
        # 동일한 사이즈 4개로 슬라이싱 : f,g,i,o
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]
        
        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)
        
        c_next = f*c_prev + g*i
        h_next = o*np.tanh(c_next)
        
        self.cache = (x,h_prev,c_prev,c_next, f, g, i, o)
        
        return h_next,c_next
    
    def backward(self,dh_next,dc_next):
        Wx, Wh, b = self.params
        x,h_prev,c_prev,c_next, f, g, i, o  = self.cache
        
        tanh_c_next = np.tanh(c_next)
        
        ds = dc_next + (dh_next * o)*(1 - tanh_c_next**2)
        
        dc_prev = ds*f
        
        df = ds * c_prev
        di = ds * g
        dg = ds * i
        do = dh_next * tanh_c_next
        
        # sigmoid 미분 : y*(1-y) 
        
        df *=  f * (1 - f)
        di *=  i * (1 - i)
        do *=  o * (1 - o)
        
        # tanh 미분 : (1-y**2) 
        dg *= (1 - g**2)
        
        dA = np.hstack((df,dg,di,do))  # 수평으로 합치기 (slice의 역전파)
        
        # Matmul 역전파
        dWh = np.dot(h_prev.T, dA)       
        dh_prev = np.dot(dA, Wh.T)
        
        # Matmul 역전파
        dWx = np.dot(x.T, dA)
        dx = np.dot(dA, Wx.T)
        
        db = dA.sum(axis=0)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        return dx, dh_prev, dc_prev
    
    
# TimeLSTM 클래스
class TimeLSTM:
    def __init__(self,Wx, Wh, b, stateful=False):
        self.params = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.layers = None             
        
        self.h,self.c = None, None  
        self.dh = None                 
        self.statuful = stateful   
        
    def forward(self,xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape          # N : batch size, T : sequence length , D : input size
        H = Wh.shape[0]             # H : hidden size,  Wh : (H,4*H)  , Wx : (D,4*H)
        
        self.layers = []
        hs = np.empty((N,T,H),dtype='f')  
        
        if not self.statuful or self.h is None:
            self.h = np.zeros((N,H), dtype='f')
            
        if not self.statuful or self.c is None:
            self.c = np.zeros((N,H), dtype='f')
            
        for t in range(T):               
            layer = LSTM(*self.params)     
            self.h , self.c = layer.forward(xs[:,t,:], self.h, self.c ) 
                                                      
            hs[:,t,:] = self.h
            self.layers.append(layer)
            
        return hs   # 출력 (N,T,H)  
    
    def backward(self,dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape  
        D = Wx.shape[0]   # Wx : (D,4*H)
        
        dxs = np.empty((N,T,D),dtype='f')
        dh , dc = 0 , 0         
        grads = [0,0,0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx,dh,dc = layer.backward(dhs[:,t,:] + dh, dc)  
            dxs[:,t,:] = dx
            
            for i,grad in enumerate(layer.grads): 
                grads[i] += grad
                
        for i,grad in enumerate(grads) :
            self.grads[i][...] = grad
        self.dh = dh
        
        return dxs   

    def set_state(self,h,c=None):
        self.h , self.c = h,c
    
    def reset_state(self):
        self.h , self.c = None, None
        

import pickle

class Rnnlm:
    def __init__(self,vocab_size=10000, wordvec_size=100,hidden_size=100 ):
        V,D,H = vocab_size, wordvec_size,hidden_size
        rn = np.random.randn
        
        # 가중치 초기화
        embed_W = (rn(V,D) / 100).astype('f')       # Embediing 계층의 출력 (N,D)
        lstm_Wx = (rn(D,4*H) / np.sqrt(D)).astype('f') # Xavier(=Glorot) Initializer, https://gomguard.tistory.com/184
        lstm_Wh = (rn(H,4*H) / np.sqrt(H)).astype('f') # Xavier(=Glorot) Initializer, https://gomguard.tistory.com/184
        lstm_b = np.zeros(4*H).astype('f')             # LSTM 계층의 출력 : (N,T,H)
        affine_W = (rn(H,V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        # 계층 생성
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx,lstm_Wh,lstm_b,stateful=True),
            TimeAffine(affine_W,affine_b)            
        ]
        self.loss_layer = TimeSoftmaxWithLoss() 
        self.lstm_layer = self.layers[1]
        
        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params,self.grads = [],[]
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self,xs):
        for layer in self.layers:
             xs = layer.forward(xs)
        return xs        
            
    def forward(self,xs,ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score,ts)
        return loss
    
    def backward(self,dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers): # 3회
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.lstm_layer.reset_state()
        
    def save_params(self, file_name='Rnnlm.pkl'):
        with open(file_name,'wb') as f:
            pickle.dump(self.params,f)
            
    def load_params(self, file_name='Rnnlm.pkl'):
        with open(file_name,'rb') as f:
            self.params = pickle.load(f)   
            
            
import sys
def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print('퍼플렉서티 평가 중 ...')
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write('\r%d / %d' % (iters, max_iters))
        sys.stdout.flush()

    print('')
    ppl = np.exp(total_loss / max_iters)
    return ppl


class BetterRnnlm():
    '''
     LSTM 계층을 2개 사용하고 각 층에 드롭아웃을 적용한 모델이다.
     아래 [1]에서 제안한 모델을 기초로 하였고, [2]와 [3]의 가중치 공유(weight tying)를 적용했다.

     [1] Recurrent Neural Network Regularization (https://arxiv.org/abs/1409.2329)
     [2] Using the Output Embedding to Improve Language Models (https://arxiv.org/abs/1608.05859)
     [3] Tying Word Vectors and Word Classifiers (https://arxiv.org/pdf/1611.01462.pdf)
    '''
    def __init__(self,vocab_size=10000, wordvec_size=650,hidden_size=650,dropout_ratio=0.5):
        V,D,H = vocab_size, wordvec_size,hidden_size
        rn = np.random.randn
        
        # 가중치 초기화
        embed_W = (rn(V,D) / 100).astype('f')  # Affine 계층과 가중치 공유    
        lstm_Wx1 = (rn(D,4*H) / np.sqrt(D)).astype('f') # Wx1 : (D,4*H)
        lstm_Wh1 = (rn(H,4*H) / np.sqrt(H)).astype('f') 
        lstm_b1 = np.zeros(4*H).astype('f')          
        lstm_Wx2 = (rn(H,4*H) / np.sqrt(D)).astype('f')  # Wx2: (H,4*H)
        lstm_Wh2 = (rn(H,4*H) / np.sqrt(H)).astype('f') 
        lstm_b2 = np.zeros(4*H).astype('f')             
        # affine_W = (rn(H,V) / np.sqrt(H)).astype('f') # 사용하지 않음
        affine_b = np.zeros(V).astype('f')
        
        # 계층 생성
        self.layers = [
            TimeEmbedding(embed_W),        # (V,D)
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1,lstm_Wh1,lstm_b1,stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2,lstm_Wh2,lstm_b2,stateful=True),  
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T,affine_b)  # embed_W.T : (D,V), D=H=650 이므로 (H,V)와 같음, weight tying!!        
        ]
        self.loss_layer = TimeSoftmaxWithLoss() 
        self.lstm_layers = [self.layers[2],self.layers[4]]
        self.drop_layers = [self.layers[1],self.layers[3],self.layers[5]]          
        
        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params,self.grads = [],[]
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self,xs, train_flag=False): 
        for layer in self.drop_layers: # 예측 시에는 dropout을 사용하지 않도록 플래그를 설정
            layer.train_flag = train_flag  
        
        for layer in self.layers:      # 7회 반복
             xs = layer.forward(xs)
        return xs        
            
    def forward(self,xs,ts, train_flag=True):
        score = self.predict(xs, train_flag) # 학습 시에는 dropout을 사용하도록 플래그를 설정
        loss = self.loss_layer.forward(score,ts)
        return loss
    
    def backward(self,dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers): # 7회
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        for layer in self.lstm_layers: # 2회
            layer.reset_state()
        
    def save_params(self, file_name='BetterRnnlm.pkl'):
        with open(file_name,'wb') as f:
            pickle.dump(self.params,f)
            
    def load_params(self, file_name='BetterRnnlm.pkl'):
        with open(file_name,'rb') as f:
            self.params = pickle.load(f)       
    
    
class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params,self.grads = [],[]
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True   # 학습시에만 True로 설정하여 dropout 적용, 예측시에는 사용하지 않음
        
    def forward(self,xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio            
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs
            
    def backward(self, dout):
        return dout * self.mask  
    
# Encoder class
class Encoder:
    def __init__(self,vocab_size, wordvec_size,hidden_size ):
        V,D,H = vocab_size, wordvec_size,hidden_size
        rn = np.random.randn
        
        # 가중치 초기화
        embed_W = (rn(V,D) / 100).astype('f')       
        lstm_Wx = (rn(D,4*H) / np.sqrt(D)).astype('f') 
        lstm_Wh = (rn(H,4*H) / np.sqrt(H)).astype('f') 
        lstm_b = np.zeros(4*H).astype('f')             
        
        # 계층 생성
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx,lstm_Wh,lstm_b,stateful=False)
        
        # 모든 가중치와 기울기를 모은다.
        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads

    def forward(self,xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)    # [N,T,H] , 3차원
        self.hs = hs
        return hs[:,-1,:]             # TimeLSTM 계층의 마지막 은닉 상태 h를 반환, [N,H], 2차원
    
    def backward(self,dh):
        dhs = np.zeros_like(self.hs)
        dhs[:,-1,:] = dh
        
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout   
    
class Seq2seq:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V,D,H = vocab_size, wordvec_size, hidden_size
        
        self.encoder = Encoder(V,D,H)
        self.decoder = Decoder(V,D,H)
        self.softmax = TimeSoftmaxWithLoss()
        
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
        
    def forward(self,xs,ts):
        decoder_xs, decoder_ts = ts[:,:-1], ts[:,1:] # decoder의 입력: 마지막 단어를 제외
                                                     # softmaxwithloss 의 입력 : 첫 단어를 제외
        
        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs,h)
        loss = self.softmax.forward(score,decoder_ts)
        return loss
    
    def backward(self,dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout
    
    def generate(self,xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h,start_id,sample_size)
        
        return sampled    
    
 # generate()을 호출하여 예측한 답이 맞는지 여부를 출력하고 맞으면 1을 틀리면 0을 반환해주는  함수
# Q 77+85  
# T 162 
# O 162 
def eval_seq2seq(model, question, correct, id_to_char,
                 verbos=False, is_reverse=False):
    correct = correct.flatten()
    # 머릿글자
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    # 문자열로 변환
    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])
    guess = ''.join([id_to_char[int(c)] for c in guess])

    if verbos:
        if is_reverse:
            question = question[::-1]

        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
        print('Q', question)
        print('T', correct)

        is_windows = os.name == 'nt'

        if correct == guess:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'O'
            print(mark + ' ' + guess)
        else:
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
            print(mark + ' ' + guess)
        print('---')

    return 1 if guess == correct else 0
        
    