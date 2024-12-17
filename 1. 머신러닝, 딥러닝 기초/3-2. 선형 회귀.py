import matplotlib.pyplot as plt
import numpy as np

from common import perch_length, perch_weight
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# k-최근접 이웃 회귀의 한계
# 훈련 세트, 테스트 세트 만들고 훈련하기
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)

# 길이가 50cm인 농어의 무게 예측하기
print(knr.predict([[50]])) # [1033.33333333]
# 하지만 실제로, 이 농어의 무게는 1033 보다 훨씬 더 나간다고 한다. 문제가 뭘까

# 50cm 농어의 이웃을 구하고, 산점도를 그려보자
distances, indexes = knr.kneighbors([[50]])
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker='D') # 이웃 샘플 표시
plt.scatter(50, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# k-근접 알고리즘은 이웃들의 평균을 구해서 도출하기 때문에, 더 큰값이 제대로 구해지지 않는다.
# 그래서 다른 알고리즘을 사용할 거다. => 선형 회귀
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# 선형 회귀 모델 훈련하기
lr.fit(train_input, train_target)
print(lr.predict([[50]])) # [1241.83860323]

# 선형 회귀가 학습한 직선은? 농어 무게 = a * 농어 길이 + b
print(lr.coef_, lr.intercept_) # a(계수, 가중치)와 b 값 확인 가능

# length 15에서 50까지 직선과 산점도 그려보기.
plt.scatter(train_input, train_target)
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 점수는?
print(lr.score(train_input, train_target)) # 0.9398463339976041
print(lr.score(test_input, test_target)) # 0.824750312331356

# 훈련세트의 점수가 높지 않다. 전체적으로 과소적합되었다. 직선이 좀 이상함. 그럼 곡선은 어떨까?
# 이차방정식을 만들기 위해 농어의 길이를 제곱한 값을 추가로 저장하기
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]])) # [1573.98423528]
print(lr.coef_, lr.intercept_) # [  1.01433211 -21.55792498] 116.0502107827827
# 구해진 식 => y = 1.01*x^2 -21.6*x + 116.05
# 다항식이 구해졌다. => 다항 회귀

# 산점도를 그려보자
point = np.arange(15, 50)
plt.scatter(train_input, train_target)
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
plt.scatter(50, 1574, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()