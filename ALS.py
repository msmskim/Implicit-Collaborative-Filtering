## Cython(C를 사용한 Python)의 implicit Alternating Least Square 알고리즘을 사용한 방식입니다.
## 관련 패키지 호출
import os
import sys
import implicit
import pandas as pd
import numpy as np
import random
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler

# 구매 데이터 로딩
def load_data():
    data = pd.read_excel('구매목록.xlsx')
    data["buying_weights"] = 1
    data = data[["goods_code", "large_cat_id", "m_id", "buying_weights"]]
    return data

# 조회수 데이터 로딩
def load_click():
    click1 = pd.read_excel("1-1.xlsx")
    click2 = pd.read_excel("1-2.xlsx")
    click = pd.concat([click1, click2])
    click = click.drop('goods_name', axis = 1)
    
    return click

# 데이터 합치기
def merging_data(data, click):
    merged_data = pd.merge(click, data, how='outer')
    merged_data = merged_data.drop_duplicates(["goods_code", "m_id"])
    
    return merged_data

### 가중치 행 만들기
def making_weights(merged_data, weights):
    merged_data.loc[pd.isnull(merged_data["cnt"]), "cnt"] = 0
    merged_data.loc[pd.isnull(merged_data["buying_weights"]), "buying_weights"] = 0
    merged_data["preference_weights"] = merged_data["cnt"] + weights*merged_data["buying_weights"]
    
    
    return merged_data

# 유저가 뭘 샀는지 unique하게 볼 수 있게 만들기
# 추후 아이템 코드 비교를 위해서 사용하게 됨
def cleaning_data(weighted_data):
    cleaned_data = weighted_data.groupby(['m_id', 'goods_code']).sum().reset_index()
    cleaned_data = cleaned_data[["m_id", "goods_code", "large_cat_id", "preference_weights"]]
    
    return cleaned_data

# 유저 - 아이템 sparse matrix 제작
def MakingSparse(cleaned_data):
    
    users = list(np.sort(cleaned_data["m_id"].unique()))
    goods = list(np.sort(cleaned_data["goods_code"].unique()))
    values = list(cleaned_data["preference_weights"])
    
    # 행과 열 정의
    cleaned_data["users"] = cleaned_data["m_id"].astype("category", categories = users).cat.codes
    cleaned_data["items"] = cleaned_data["goods_code"].astype('category', categories = goods).cat.codes
    
    # 희소행렬
    item_user_matrix = sparse.csr_matrix((values, (cleaned_data['items'], cleaned_data["users"])), shape = (len(goods), len(users)))
    user_item_matrix = sparse.csr_matrix((values, (cleaned_data["users"], cleaned_data["items"])), shape = (len(users), len(goods)))
    
    return item_user_matrix, user_item_matrix, cleaned_data

## Sparsity 확인, ALS 알고리즘의 경우 99.5%의 sparsity까지 커버 가능함
matrix_size = item_user_matrix.shape[0]*item_user_matrix.shape[1]
num_purchases = len(item_user_matrix.nonzero()[0])
sparsity = 100*(1-(num_purchases/matrix_size))
sparsity

### implicit 알고리즘
def Implementing_ALS(factors, regularization, n_iters, alpha):
    model = implicit.als.AlternatingLeastSquares(factors = factors, regularization = regularization, iterations = n_iters)
    alpha_val = alpha
    data_conf = (item_user_matrix * alpha_val).astype('double')

    return model, data_conf

## implicit 알고리즘 적용 후 similar한 아이템 코드 10개 반환, 아이템 기반의 추천에 사용
def SimilarItems(model, data_conf, item_id, n_similar, cleaned_data):
    model.fit(data_conf)
    similar = model.similar_items(item_id, n_similar)
    
    ## item_id의 원래 아이템 코드
    origin_code = cleaned_data["goods_code"].loc[cleaned_data["items"] == item_id].iloc[0]
    
    ## Similar 아이템 원래의 코드 가져오기
    goods = []
    scores = []
    # items와 scores dict
    for item in similar:
        goods.append(item[0])
        scores.append(item[1])

    ## 실제 코드 저장
    code = []
    for num in goods:
        code.append(cleaned_data["goods_code"].loc[cleaned_data["items"] == num].iloc[0])
    
    # 추천 아이템과 점수 사전으로 반환하기
    similar_item = {}
    for i in range(len(goods)):
        similar_item[code[i]] = scores[i]
    
    result = {}
    result[origin_code] = similar_item
    
    return result

##  User에게 추천
def UserbasedRecommender(user_id, user_item_matrix, n_items, cleaned_data):
    recommended = model.recommend(user_id, user_item_matrix, n_items)
    
    # origin은 원래 user_id
    origin = cleaned_data["m_id"].loc[cleaned_data["users"] == user_id].iloc[0]
    
    # 추천 goods와 score 사전으로 저장하기
    ## goods와 scores 각각 리스트 만들기
    goods = []
    scores = []
    
    for element in recommended:
        goods.append(element[0])
        scores.append(element[1])
        
    ## 실제 goods_code 리스트에 저장하기
    codes = []
    
    for num in goods:
        codes.append(cleaned_data["goods_code"].loc[cleaned_data["items"] == num].iloc[0])
    
    ## 추천 아이템과 점수 사전으로 반환하기
    similar_items = {}
    for i in range(len(codes)):
        similar_items[codes[i]] = scores[i]
        
    
    result = {}
    result[origin] = similar_items
    
    return result
    
# 결과 확인
def main():
    data = load_data()
    click = load_click()
    merged_data = merging_data(data, click)
    weighted_data = making_weights(merged_data, weights = 10)
    cleaned_data = cleaning_data(weighted_data)
    item_user_matrix, user_item_matrix, cleaned_data = MakingSparse(cleaned_data)
    model, data_conf = Implementing_ALS(factors = 20, regularization = 0.1, n_iters = 100, alpha = 40)
    item_result = SimilarItems(model, data_conf, 1234, 10, cleaned_data)
    item_result
    recommended_result = UserbasedRecommender(1234, user_item_matrix, 10, cleaned_data)
    recommended_result
    
main()
