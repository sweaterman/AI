# 사실 농어의 길이 말고 높이와 두께 데이터도 있었다..!
# 선형 회귀는 특성이 많을 수록 효과가 잘 나온다.
# 특성 공학: 새로운 특성을 뽑아내는 작업
import pandas as pd
from common import perch_weight
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('https://bit.ly/perch_csv')
perch_full = df.to_numpy()
# print(perch_full)
train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42)

# 사이킷런의 변환기: 특성을 만들거나 전처리하기 위한 다양한 클래스를 제공
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(poly.get_feature_names_out()) # ['x0' 'x1' 'x2' 'x0^2' 'x0 x1' 'x0 x2' 'x1^2' 'x1 x2' 'x2^2']

# 테스트 세트를 변환하자. (훈련세트로 학습한 변환기를 사용해, 테스트 세트까지 변환해야한다.)
test_poly = poly.transform(test_input)

# 이제 훈련해보자
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target)) # 0.9903183436982126
print(lr.score(test_poly, test_target)) # 0.9714559911594125

# 과소 적합 문제는 해결한 것 같다.
# 특성을 더 많이 추가해서 5제곱까지 늘리면 어떻게 될까?
poly = PolynomialFeatures(include_bias=False, degree=5)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
lr.fit(train_poly, train_target)

print(lr.score(train_poly, train_target)) # 0.9999999999997232
print(lr.score(test_poly, test_target)) # -144.40564483377855
# 특성의 개수를 크게 늘리면 훈련 세트에 대해 거의 완벽하게 학습되어, 훈련 세트에 너무 과대적합 된다.

# 규제: 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 훼방하는 것
# 선형 회귀 모델의 경우 특성에 곱해지는 계수(또는 기울기)의 크기를 작게 만드는 일
# 규제를 적용하기 전, 먼저 정규화를 해야한다!
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 선형 회귀 모델에 규제를 추가한 모델: 릿지(ridge), 라쏘(lasso)
# 릿지: 계수를 제곱한 값을 기준으로 규제를 적용
# 라쏘: 계수의 절댓값을 기준으로 규제를 적용
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target)) # 0.9896101671037343
print(ridge.score(test_scaled, test_target)) # 0.9790693977615387
# 많은 특성을 사용했음에도 불구하고, 훈련세트에 과대적합되지 않아 테스트 세트에서도 좋은 성능을 내고 있다.

# 모델 객체를 만들 때 alpha 매개변수로 규제의 강도를 조절한다.
# alpha 값이 크면 규제 강도가 세지고 과소적합되도록 유도
# alpha 값이 작으면 계수를 줄이는 역할이 줄어들고 선형회귀와 유사해지므로 과대적합될 가능성이 큼
# 이렇듯 사람이 알려줘야 하는 파라미터 => 하이퍼파라미터

# 적절한 alpha 값을 찾는 방법 => R^2의 그래프를 그려본다. 훈련세트와 테스트 세트의 점수가 가장 가까운 지점이 최적의 alpha 값이다.
# alpha 값을 0.001 부터 10배씩 늘려가며 결과를 확인해보자
import matplotlib.pyplot as plt
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))
# 그래프로 확인해보자
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# 적절한 alpha값은 두 그래프가 가장 가깝고 테스트의 점수가 가장 높은 -1, 즉 0.1이다
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target)) # 0.9903815817570367
print(ridge.score(test_scaled, test_target)) # 0.9827976465386928

# 라쏘 회귀
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target)) # 0.989789897208096
print(lasso.score(test_scaled, test_target)) # 0.9800593698421883

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    lasso = Lasso(alpha=alpha, max_iter=10000) # 지정한 반복 횟수가 부족하면 경고가 나오기 때문에 설정해줌
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# 적절한 alpha 값은 1, 즉 10이다.
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target)) # 0.9888067471131867
print(lasso.score(test_scaled, test_target)) # 0.9824470598706695