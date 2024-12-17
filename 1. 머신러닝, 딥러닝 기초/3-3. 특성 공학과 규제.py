# 사실 농어의 길이 말고 높이와 두께 데이터도 있었다..!
# 선형 회귀는 특성이 많을 수록 효과가 잘 나온다.
# 특성 공학: 새로운 특성을 뽑아내는 작업
import pandas as pd
from common import perch_weight
from sklearn.model_selection import train_test_split

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

# 160
