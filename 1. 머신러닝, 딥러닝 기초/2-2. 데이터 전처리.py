from common import fish_length, fish_weight
import numpy as np
from sklearn.model_selection import train_test_split

# 넘파이 column_stack 을 사용해, 길이 + 무게를 한 배열에 집어넣어 만들어보기
fish_data = np.column_stack((fish_length, fish_weight))
#print(fish_data)

# 넘파이 concatenate() 를 사용해 타깃 데이터 만들기
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
print(fish_target)

# 사이킷런의 train_test_split() 함수로 리스트나 배열을 훈련 세트와 테스트 세트로 알잘딱 나눠줌
# 기본적으로 25%를 테스트 세트로 떼어낸다.
# 하지만 무작위로 섞였을 때 샘플링 편향이 나타날 수도 있음. 그래서 사용하는 것이 stratify 매개변수
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
