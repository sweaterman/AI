from common import fish_data, fish_target
from common import kn, np

# 지도 학습 = 훈련을 위한 훈련 데이터(데이터(입력) + 정답(타깃))가 필요 => 정답을 맞히는 것을 학습
# 비지도 학습 = 타깃 없이 입력 데이터만 사용 => 정답을 맞히진 않지만, 데이터 파악 및 변형에 도움을 줌
# 강화 학습 = 타깃이 아니라 알고리즘이 행동한 결과로 얻은 보상을 사용해 학습된다. (여기서는 안 다룸)

# 테스트 세트: 평가에 사용하는 데이터
# 훈련 세트: 훈련에 사용하는 데이터


# 샘플: 하나의 생선 데이터
# 그래서 총 49개의 데이터가 있다.
# 처음 35개를 훈련 세트로, 나머지 14개를 테스트 세트로 사용해보자
train_input = fish_data[:35]
train_target = fish_target[:35]
test_input = fish_data[35:]
test_target = fish_target[35:]

kn = kn.fit(train_input, train_target)
# print(kn.score(test_input, test_target)) # 샘플링 편향 때문에 정확도가 0이 나오게 된다.

# 넘파이 배열로 변환
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
# print(input_arr)
# print(input_arr.shape) # 샘플 수, 특성 수 출력

# 샘플 세트 섞기
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# 잘 섞였는 지 확인해보자