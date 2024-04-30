# """
# 환경 모듈
# """

class Environment:
    PRICE_IDX = 4  # csv 파일에서, 종가의 index(열) 위치

    def __init__(self, chart_data=None):
        """
        chart_data : 차트 데이터
        """
        self.chart_data = chart_data # 주식 종목의 차트 데이터
        self.observation = None # 현재 관측치
        self.idx = -1 # 관측치의 현재 인덱스 번호

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        """
        데이터를 순차적으로 불러옴
        ex) 처음 불러오면 1번쨰 열 데이터를 반환
            두번째로 불러오면 2번째 열 데이터를 반환
        """

        # 더 가져올 데이터가 있는지 확인
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx] # chart_data의 self.idx 인덱스의 데이터를 observation에 저장
            return self.observation
        return None

    def get_price(self):
        """
        현재 관측지를 기준으로 종가를 불러오는 함수
        """
        if self.observation is not None:
            return self.observation.iloc[self.PRICE_IDX] 
        return None
