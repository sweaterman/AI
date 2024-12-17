import matplotlib.pyplot as plt
from common import perch_length, perch_weight
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# 농어의 길이, 높이, 두께를 측정한 데이터로 무게를 예측해보자.
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 먼저 훈련 세트와 테스트 세트로 나누기
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)

# 사이킷런에 사용할 훈련세트는 2차원 배열을 사용해야한다.
# 그런데 앞에서와는 달리, 특성 한가지만을 사용하기 때문에 강제로 2차원 배열로 만들어주기
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target)) # 점수는 0.9928094061010639

# 결정계수 (R^2)
# 1- ((타깃-예측)^2의 합 / (타깃-평균)^2의 합)
# 1에 가까울 수록 좋다.
test_prediction = knr.predict(test_input) # 테스트 세트에 대한 예측 만들기
mae = mean_absolute_error(test_target, test_prediction)
print(mae) # 결과: 19.157142857142862 => 예측이 평균적으로 19g 정도 타깃값과 다르다.

# 앞에서 훈련한 모델을 사용해 "훈련 세트"의 R^2 점수를 확인해보자.
print(knr.score(train_input, train_target)) # 점수는 0.9698823289099255

# 훈련 세트 점수 > 테스트 세트 점수 : 과대 적합 => 모델을 덜 복잡하게 만들어야 한다.
# 훈련 세트 점수 < 테스트 세트 점수 : 과소 적합 => 모델을 더 복잡하게 만들어야 한다.
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target)) # 점수는 0.9804899950518966

