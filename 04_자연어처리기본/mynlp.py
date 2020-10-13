#!/usr/bin/env python
# coding: utf-8

# ##  통계기반 기법

# #### 말뭉치 또는 코퍼스(corpus)는 자연어 연구를 위해 특정한 목적을 가지고 언어의 표본을 추출한 집합이다.
# #### 대량의 텍스트 데이터
# 컴퓨터의 발달로 말뭉치 분석이 용이해졌으며 분석의 정확성을 위해 해당 자연어를 형태소 분석하는 경우가 많다. 확률/통계적 기법과 시계열적인 접근으로 전체를 파악한다. 언어의 빈도와 분포를 확인할 수 있는 자료이며, 현대 언어학 연구에 필수적인 자료이다

# # In[1]:


# text = 'You say goodbye and I say hello.' 


# # In[2]:


# # 소문자로 변환
# text = text.lower()
# text


# # In[3]:


# text = text.replace('.',' .')
# text


# # In[4]:


# words = text.split(' ')  
# words # 단어 목록으로 변환


# # In[5]:


# list(set(words)) # 중복된 단어를 제거해야할 경우


# # ###  딕셔너리를 이용하여 단어 ID와 단어를 짝지어 주는 대응표 작성

# # In[6]:


# word_to_id = {}   # dict , {'you':0, 'say':1,.....}
# id_to_word = {}   # dict , {0:'you', 1:'say',.....}

# for word in words: # 8회
#     print(word)
#     if word not in word_to_id:    # 중복 방지
#         new_id = len(word_to_id)  # 0 ~ 6
#         word_to_id[word] = new_id # word_to_id['you'] = 0
#         id_to_word[new_id] = word # id_to_word[0] = 'you'


# # In[7]:


# word_to_id


# # In[8]:


# id_to_word


# # In[9]:


# word_to_id['hello']


# # In[10]:


# id_to_word[2]


# # ### corpus를 숫자 벡터로 변환

# # In[11]:


# # text = 'You say goodbye and I say hello.' 
# # ['you', 'say', 'goodbye', 'and', 'i', 'say', 'hello', '.']
# # [  0,    1,       2,       3,     4,    1,    5,       6 ]  
import numpy as np
# corpus = [word_to_id[w] for w in words]
# corpus = np.array(corpus)
# print(corpus)


# ###  말뭉치를 이용하기 위한 전처리 함수 구현

# In[12]:


def preprocess(text):
    text = text.lower()
    text = text.replace('.',' .')
    words = text.split(' ')
    
    word_to_id = {}
    id_to_word = {}

    for word in words: 
        if word not in word_to_id:    
            new_id = len(word_to_id)  
            word_to_id[word] = new_id 
            id_to_word[new_id] = word 
            
    corpus = np.array([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word


# # In[13]:


# text = 'You say goodbye and I say hello.' 
# corpus, word_to_id, id_to_word = preprocess(text)


# # In[14]:


# corpus


# # In[15]:


# word_to_id


# # In[16]:


# id_to_word


# # In[17]:


# my_text = 'The cat ran very fast and the mouse could not run away.'
# corpus, word_to_id, id_to_word = preprocess(my_text)
# corpus


# # In[18]:


# word_to_id  # 12개 단어 사용


# # In[19]:


# id_to_word


# # ### 분포가설(distibutional hypothesis) : 단어의 의미는 주변 단어에 의해 형성된다
# # 
# # ### 동시 발생 행렬(Co-occurence Matrix)
# # : 모든 단어의 동시 발생 횟수를 벡터로 표현한 행렬
# # 
# # - 주변 단어(맥락,Contexts)
# # - 중간 단어(타깃,target)
# # - 윈도 사이즈 : 중간단어를 기준으로 앞뒤로 사용할 주변 단어 갯수

# # In[20]:


# text = 'You say goodbye and I say hello.' 
# corpus, word_to_id, id_to_word = preprocess(text)


# # In[21]:


# corpus


# # In[22]:


# id_to_word


# # In[23]:


# C = np.array([
#     [0, 1, 0, 0, 0, 0, 0],
#     [1, 0, 1, 0, 1, 1, 0],
#     [0, 1, 0, 1, 0, 0, 0],
#     [0, 0, 1, 0, 1, 0, 0],
#     [0, 1, 0, 1, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 1, 0],
# ], dtype=np.int32)


# # In[24]:


# C


# # In[25]:


# C[0] # id 가 0인 단어('you')의 벡터 표현


# # In[26]:


# C[4] # id 가 4인 단어('i')의 벡터 표현


# # In[27]:


# 동시 발생 행렬을 생성하는 함수 구현
def create_co_matrix(corpus, vocab_size, windows_size=1):
    corpus_size = len(corpus)  # 8
    co_matrix = np.zeros((vocab_size,vocab_size), dtype=np.int32) # 2차원, (7,7)
    # import pdb; pdb.set_trace()
    for idx, word_id in enumerate(corpus): # 8회 반복, idx : 0 to 7 , word_id: [0, 1, 2, 3, 4, 1, 5, 6]
        for i in range(1, windows_size + 1) : # 1회,  windows_size=1인경우 i는 항상 1
            left_idx = idx - i
            right_idx = idx + i
            
            if left_idx >=0 :   # 처음 시작 단어 제외
                left_word_id = corpus[left_idx]
                co_matrix[word_id,left_word_id] += 1
                
            if right_idx < corpus_size: #마지막 단어 제외
                right_word_id = corpus[right_idx]
                co_matrix[word_id,right_word_id] += 1
    return co_matrix


# # In[28]:


# text = 'You say goodbye and I say hello.' 
# corpus, word_to_id, id_to_word = preprocess(text)
# print(corpus)
# print(word_to_id)
# vocab_size = len(word_to_id)
# C = create_co_matrix(corpus,vocab_size)
# C


# # In[29]:


# text = 'I like apple and you like banana.'
# corpus, word_to_id, id_to_word = preprocess(text)
# print(corpus)
# print(word_to_id)
# vocab_size = len(word_to_id)
# C = create_co_matrix(corpus,vocab_size)
# C


# # ## 벡터 간 유사도 
# # 
# # ####  유사도(Similarity)
# # https://goofcode.github.io/similarity-measure
# # 
# # #### norm : 벡터의 크기
# # https://bskyvision.com/825
# # 
# # ### 코사인 유사도(Cosine Similarity)
# # : 1이면 완전 동일, -1 이면 완전 반대, L2 norm사용
# # 
# # ![image](https://goofcode.github.io/assets/img/%E1%84%8B%E1%85%A7%E1%84%85%E1%85%A5%20%E1%84%80%E1%85%A1%E1%84%8C%E1%85%B5%20%E1%84%8B%E1%85%B2%E1%84%89%E1%85%A1%E1%84%83%E1%85%A9%20%E1%84%8E%E1%85%B3%E1%86%A8%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%87%E1%85%A5%E1%86%B8%20(Similarity%20Measure)/download.png)

# # In[30]:


# def cos_similarity(x,y): # 코싸인 유사도
#     nx = x / np.sqrt(np.sum(x**2))
#     ny = y / np.sqrt(np.sum(y**2))
#     return np.dot(nx,ny) 
# # 입력 인수로 제로 벡터(원소가 모두 0인 벡터)가 들어오면 'divide by zero' 오류 발생


# # In[31]:


# 개선된 코싸인 유사도 : 작은 값 eps(엡실론)을 분모에 더해준다, 부동소수점 계산시 반올림되어 다른 값에 흡수된다
def cos_similarity(x,y,eps=1e-8): # 코싸인 유사도
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx,ny) 


# In[32]:


# Euclidean Distance : 0이면 완전동일
def eucl_dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

# doc1 = np.array([2,3,0,1])
# doc2 = np.array([1,2,3,1])
# doc3 = np.array([2,1,2,2])
# doc4 = np.array([1,1,0,1])
# docQ = np.array([1,1,0,1])

# print(eucl_dist(doc1,docQ))
# print(eucl_dist(doc2,docQ))
# print(eucl_dist(doc3,docQ))
# print(eucl_dist(doc4,docQ))


# In[33]:


# text = 'You say goodbye and I say hello.' 
# corpus, word_to_id, id_to_word = preprocess(text)
# # print(corpus)
# print(word_to_id)
# vocab_size = len(word_to_id)
# C = create_co_matrix(corpus,vocab_size)
# # print(C)

# c0 = C[word_to_id['you']] # 'you' 단어의 벡터, [0 1 0 0 0 0 0]
# c1 = C[word_to_id['i']]   # 'i' 단어의 벡터,   [0 1 0 1 0 0 0]
# print('you:',c0)
# print('i:  ',c1)
# print(cos_similarity(c0,c1))
# print(eucl_dist(c0,c1))


# # In[34]:


# text = 'I like apple and you like banana.'
# corpus, word_to_id, id_to_word = preprocess(text)
# # print(corpus)
# # print(word_to_id)
# vocab_size = len(word_to_id)
# C = create_co_matrix(corpus,vocab_size)

# c0 = C[word_to_id['apple']]
# c1 = C[word_to_id['banana']]   
# print('apple:',c0)
# print('banana:  ',c1)
# print(cos_similarity(c0,c1))
# print(eucl_dist(c0,c1))


# ### 유사 단어의 랭킹 표시

# In[35]:


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:   # 동일단어는 제외
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


# In[36]:


# x = np.array([100, -20, 2])
# x.argsort() # 오름 차순
# (-x).argsort() # 내림 차순


# # In[37]:


# text = 'You say goodbye and I say hello.' 
# corpus, word_to_id, id_to_word = preprocess(text)
# # print(corpus)
# print(word_to_id)
# vocab_size = len(word_to_id)
# C = create_co_matrix(corpus,vocab_size)
# most_similar('you',word_to_id,id_to_word, C,top=5)


# # In[38]:


# most_similar('hello',word_to_id,id_to_word, C,top=5)


# # In[ ]:




