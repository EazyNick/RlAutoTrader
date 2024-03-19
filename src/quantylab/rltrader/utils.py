import time
import datetime
import numpy as np


# 날짜, 시간 관련 문자열 형식
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"


def get_today_str():
    # datetime.date.today()를 사용하여 현재 로컬 날짜를 가져옵니다.
    # 가능한 최소 시간(datetime.datetime.min.time())과 결합하여 기본적으로 시간을 00:00:00으로 설정
    today = datetime.datetime.combine(
        datetime.date.today(), datetime.datetime.min.time())
    
    today_str = today.strftime('%Y%m%d')
    return today_str

# Format_DATETIME에 의해 정의된 형식으로 현재 datetime을 문자열로 반환
def get_time_str():
    # time.time()을 호출하여 에포크 이후 현재 시간(초)을 가져온 다음 이 플로트를 정수로 변환
    # dattime.datetime.fromtimestamp()를 사용하여 이 타임스탬프를 dattime 개체로 변환
    # datetime 개체를 "%Y%m%d%H%M%S" 형식에 따라 문자열로 포맷하고 반환
    return datetime.datetime.fromtimestamp(
        int(time.time())).strftime(FORMAT_DATETIME) 

# 이 함수는 기계 학습과 통계, 특히 로지스틱 회귀와 신경망에서 널리 사용되는 시그모이드 함수를 계산합니다. 
# 시그모이드 함수는 실수 값을 0과 1 사이의 범위로 매핑하여 숫자를 확률로 변환하는 데 유용
# 지수 함수의 오버플로 문제를 방지하기 위해 -10에서 10 사이의 x를 클램핑합니다. 이것은 수치 안정성을 위한 실용적인 고려 사항
def sigmoid(x):
    x = max(min(x, 10), -10)
    return 1. / (1. + np.exp(-x))
