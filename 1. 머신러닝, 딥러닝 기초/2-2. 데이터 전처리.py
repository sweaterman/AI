from pandas.core import indexes

from common import fish_length, fish_weight
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 넘파이 column_stack 을 사용해, 길이 + 무게를 한 배열에 집어넣어 만들어보기
fish_data = np.column_stack((fish_length, fish_weight))
#print(fish_data)

# 넘파이 concatenate() 를 사용해 타깃 데이터 만들기
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
#print(fish_target)

# 사이킷런의 train_test_split() 함수로 리스트나 배열을 훈련 세트와 테스트 세트로 알잘딱 나눠줌
# 기본적으로 25%를 테스트 세트로 떼어낸다.
# 하지만 무작위로 섞였을 때 샘플링 편향이 나타날 수도 있음. 그래서 사용하는 것이 stratify 매개변수
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
print(kn.score(test_input, test_target)) # 1이 출력됨

print(kn.predict([[25, 150]])) # 얘는 왜 0이 나올까?

# 이웃 5개를 보자
distances, indexes = kn.kneighbors([[25,150]])
# plt.scatter(test_input[:, 0], test_input[:, 1])
# plt.scatter(25, 150, marker='^')
# plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
# plt.xlabel('length')
# plt.ylabel('weight')
#plt.show()
# 가까운 4개가 빙어라서 빙어라고 예측하는 것

# distance를 확인해보면..
print(distances)

# x축은 10~40의 범위이고, y축은 0~1000의 범위이다.
# x축을 y축과 동일하게 0~1000의 범위로 맞춰보면?
#plt.xlim((0, 1000))
#plt.show()

# 결론. 두 특성의 스케일이 다르다.
# 이런 알고리즘들은 샘플 간의 거리에 영향을 많이 받으므로, 일정한 기준으로 맞춰줄 필요가 있다.
# => 데이터 전처리

# 가장 널리 사용하는 방법은 '표준점수' (각 특성값이 0에서 표준편차의 몇 배만큼 떨어져 있는지?)
mean = np.mean(train_input, axis=0) # 평균 계산
std = np.std(train_input, axis=0)  # 표준편차 계산
train_scaled = (train_input - mean) / std # 표준점수 계산  => 넘파이의 브로드캐스팅 기능

# 이제 다시 그려보자. (샘플에도 mean, std를 적용시켜 변환해야한다.)
new = ([25, 150] - mean) / std
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# => x축과 y축의 범위가 -1.5 ~ 1.5 사이로 바뀌었다.

# 훈련하기
kn.fit(train_scaled, train_target)

# 이제 테스트를 해보자. 테스트를 할 때도 훈련세트의 평균과 표준편차로 변환해야한다.
test_scaled = (test_input - mean) / std
print(kn.score(test_scaled, test_target))
