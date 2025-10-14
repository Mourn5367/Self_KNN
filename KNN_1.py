import math
from collections import Counter
import matplotlib.pyplot as plt  # 시각화 라이브러리 import

class KnnClass:

    def __init__(self):
        self.sample_data = []
        self.labels = []
        self.k = 0
        self.new_point = []
        self.show_columns = ()
        self.dist_info_list = [] # (거리, 좌표, 레이블)을 저장할 리스트

    def init_data(self, sample_data:list, labels:list, k:int, new_point:list, show_columns:tuple):
        self.sample_data = sample_data
        self.labels = labels
        self.k = k
        self.new_point = new_point
        self.show_columns = show_columns

        # 데이터 초기화 후 바로 계산 및 예측까지 수행
        self.cal_dist()
        prediction = self.determine_knn()
        print(f"새로운 데이터 {self.new_point}의 예측 클래스는 '{prediction}' 입니다.")
        
        self.visualize_knn()

    def cal_dist(self):
        """
        거리 계산 시 (거리, 원본 좌표, 레이블) 형태로 저장하도록 수정
        """
        for i, sample_point in enumerate(self.sample_data):
            dist = self.euclidean_distance(sample_point, self.new_point)
            # 튜플에 sample_point를 추가하여 좌표 정보 저장
            self.dist_info_list.append((dist, sample_point, self.labels[i]))
        
        self.dist_info_list.sort(key=self.first_item_return)

    def determine_knn(self):
        # 상위 k개의 이웃 선택
        k_nearest_neighbors = self.dist_info_list[:self.k]

        # 이웃들의 레이블만 추출 (item[2]가 레이블)
        neighbor_labels = [item[2] for item in k_nearest_neighbors]

        # 다수결 투표
        most_common = Counter(neighbor_labels).most_common(1)
        return most_common[0][0]
    
    def visualize_knn(self):
        """
        showColumn에 지정된 피쳐를 기준으로 KNN 결과를 시각화합니다.
        """
        # 1. 시각화에 사용할 x, y축 인덱스 결정
        # 사용자가 1, 2로 입력하면 인덱스는 0, 1이 됩니다.
        x_index = self.show_columns[0] - 1
        y_index = self.show_columns[1] - 1

        # 2. 클래스별로 색상을 다르게 지정하기 위한 준비
        unique_labels = sorted(list(set(self.labels)))
        colors = plt.cm.rainbow([i/len(unique_labels) for i in range(len(unique_labels))])
        color_map = {label: color for label, color in zip(unique_labels, colors)}

        # 3. 전체 샘플 데이터 그리기
        for point, label in zip(self.sample_data, self.labels):
            plt.scatter(point[x_index], point[y_index], color=color_map[label], label=label if label in unique_labels else None)
            unique_labels.remove(label) if label in unique_labels else None # 범례 중복 방지

        # 4. 새로운 데이터 포인트 그리기 (크고 검은 별)
        plt.scatter(self.new_point[x_index], self.new_point[y_index], color='black', marker='*', s=200, edgecolor='white', label='New Point')

        # 5. K개의 이웃들 강조하기 (녹색 원으로 표시)
        k_nearest_neighbors = self.dist_info_list[:self.k]
        for dist, point, label in k_nearest_neighbors:
            plt.scatter(point[x_index], point[y_index], facecolors='none', edgecolors='green', s=150, linewidths=2)

        # 6. 그래프 꾸미기
        plt.title(f'KNN Visualization (k={self.k})')
        plt.xlabel(f'Feature {x_index + 1}')
        plt.ylabel(f'Feature {y_index + 1}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def euclidean_distance(self, sampleD:list, inputD:list):
        dist_sq = 0
        for i in range(len(sampleD)):
            dist_sq += (sampleD[i] - inputD[i])**2
        return math.sqrt(dist_sq)
    
    def first_item_return(self, _tuple:tuple):
        return _tuple[0]

def get_data_from_terminal():
    """터미널에서 사용자 입력을 받아 KNN 분석에 필요한 모든 데이터를 반환합니다."""
    
    # K 값과 새로운 데이터 입력 부분은 동일
    while True:
        try:
            k = int(input("사용할 K값을 입력하세요: "))
            break
        except ValueError:
            print("오류: 숫자를 입력해야 합니다.")

    while True:
        try:
            raw_input = input("분류할 새로운 데이터 좌표를 입력하세요 (예: 5,11): ")
            new_point = [float(x) for x in raw_input.split(',')]
            break
        except (ValueError, IndexError):
            print("오류: '숫자,숫자' 형식으로 올바르게 입력해주세요.")

    # 학습 데이터 입력 부분 수정
    print("\n학습 데이터를 입력하세요 (예: 4,12,C). 입력을 마치려면 '끝'을 입력하세요.")
    sample_data = []
    labels = []
    while True:
        raw_input = input(f"데이터 {len(sample_data) + 1}: ").strip()
        if raw_input.lower() in ['끝', 'done', 'exit']:
            if not sample_data:
                print("오류: 최소 하나 이상의 학습 데이터가 필요합니다.")
                continue
            break
        
        try:
            # 쉼표로 한 번에 나눔
            parts = raw_input.split(',')
            if len(parts) < 2: # 최소 좌표 1개 + 레이블 1개는 있어야 함
                raise ValueError("입력 형식이 잘못되었습니다.")

            # 마지막 요소를 제외한 모든 것을 좌표로 변환
            coords = [float(x) for x in parts[:-1]]
            
            # 마지막 요소를 레이블로 사용 (앞뒤 공백 제거)
            label = parts[-1].strip()
            
            sample_data.append(coords)
            labels.append(label)
        except (ValueError, IndexError):
            # 오류 메시지 수정
            print("오류: '좌표,좌표,레이블' 형식(예: 4,12,C)으로 올바르게 입력해주세요.")

    return sample_data, labels, k, new_point

def get_visualization_columns(sample_data: list) -> tuple:
    """
    학습 데이터를 기반으로 피쳐 개수를 확인하고,
    사용자에게 시각화할 두 축을 입력받아 튜플로 반환합니다.
    """
    if not sample_data:
        return ()

    feature_count = len(sample_data[0])
    show_cols = ()
    
    if feature_count >= 2:
        print(f"\n데이터의 특징(피쳐)이 {feature_count}개 있습니다.")
        raw_cols = input("시각화에 사용할 두 축의 번호를 입력하세요 (예: 1,2 / 생략하려면 Enter): ")
        try:
            # 입력값이 있을 때만 처리
            if raw_cols:
                cols = tuple(int(c) for c in raw_cols.split(','))
                if len(cols) == 2:
                    show_cols = cols
                else:
                    print("오류: 축 번호는 2개를 입력해야 합니다.")
        except ValueError:
            print("오류: 숫자를 쉼표로 구분하여 입력해주세요.")
    
    return show_cols
# --- 실행 부분 ---
if __name__ == "__main__":

    knn = KnnClass()

    # 데이터 입력
    sam, lable, k, input_d = get_data_from_terminal()

    # 시각화 할 피쳐 입력
    show_cols = get_visualization_columns(sam)

    # 시작
    knn.init_data(sam, lable, k, input_d, show_cols)
