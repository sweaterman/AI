from common import bream_length, bream_weight, smelt_length, smelt_weight
from common import plt, kn

# 산점도: x, y축으로 이뤄진 좌표계에 두 변수 (x, y)의 관계를 표현하는 방법
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
# plt.show() # 도미의 결과 => "선형"적이다!

# k-최근접 이웃 알고리즘
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
fish_data = [[l, w] for l, w in zip(length, weight)]

# 도미와 빙어를 1과 0으로 구분해준다.
fish_target = [1] * len(bream_length) + [0] * len(smelt_length)

# 훈련 시키기
kn.fit(fish_data, fish_target)
# print(kn.score(fish_data, fish_target)) # 훈련 평가

# 새로운 데이터를 주고 어떤 답을 내놓는지 보자.
# print(kn.predict([[30, 600]]))

# 단점은 데이터가 아주 많은 경우 사용하기 어렵다.
# 데이터가 크기 때문에 메모리가 많이 필요하고 직선거리를 계산하는 데도 많은 시간이 필요하다.

# print(kn._fit_X) # fish_data를 전부 갖고 있음.
# print(kn._y) # fish_target을 전부 갖고 있음.

# KNeighborsClassifier 클래스의 기본값은 5다. 가까운 5개의 데이터를 참고한다는 뜻.
from sklearn.neighbors import KNeighborsClassifier
kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)
print(kn49.score(fish_data, fish_target)) # 평가 결과 => 0.7142857142857143




