## ALS 논문 알고리즘
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import scipy.linalg.spsolve as spsolve

def ALS(train_set, lambda_val = 0.1, alpha = 40, iterations = 10, rank_size = 20, seed = 0):
    
    ### 신뢰행렬 정의
    conf = (alpha*train_set)
    
    num_user = conf.shape[0]
    num_item = conf.shape[1]
    
    ## seed 값을 정해서 Random한 X,Y feature 벡터로 시작하기 X = User_Vector, Y = Item_Vector
    rstate = np.random.RandomState(seed)
    
    ## normal distribution 안에 있는 샘플들을 추출시켜줌
    X = sparse.csr_matrix(rstate.normal(size = (num_user, rank_size)))
    Y = sparse.csr_matrix(rstate.normal(size=  (num_item, rank_size)))
    
    X_eye = sparse.eye(num_user)
    Y_eye = sparse.eye(num_item)
    lambda_eye = lambda_val * sparse.eye(rank_size)
    
    ## iterations start
    for iter in range(iterations):
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)
    ## user vec    
        for u in range(num_user):
            conf_samp = conf[u, :].toarray()
            pref = conf_samp.copy()
            
            ## 메모리 에러 ㅠㅠ
            pref[pref == 1] = 0
            pref[pref == 2] = 0
            pref[pref == 3] = 0
            pref[pref == 4] = 0
            pref[pref == 5] = 0
            pref[pref == 6] = 0
            pref[pref == 7] = 0
            pref[pref == 8] = 0
            pref[pref == 9] = 0
            pref[pref >= 10] = 1
            
            ## diag와 X, Y 정의하고 식 세우기
            CuI = sparse.diags(conf_samp, [0])
            yTCuIY = Y.T.dot(CuI).dot(Y)
            yTCupu = Y.T.dot(CuI + Y_eye).dot(pref.T)
            
            X[u] = spsolve(yTy + yTCuIY + lambda_eye, yTCupu)
            
        ## item vec        
        for i in range(num_item):
            conf_samp = conf[:, i].T.toarray()
            pref = conf_samp.copy()
            pref[pref == 1] = 0
            pref[pref == 2] = 0
            pref[pref == 3] = 0
            pref[pref == 4] = 0
            pref[pref == 5] = 0
            pref[pref == 6] = 0
            pref[pref == 7] = 0
            pref[pref == 8] = 0
            pref[pref == 9] = 0
            pref[pref >= 10] = 1
            
            CiI = sparse.diags(conf_samp, [0])
            xTCiIX = X.T.dot(CiI).dot(X)
            xTCiPi = X.T.dot(CiI + X_eye).dot(pref.T)
            Y[i] = spsolve(xTx + xTCiIX + lambda_eye, xTCiPi)

    return X, Y.T
