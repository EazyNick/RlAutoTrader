# """
# 에이전트 모듈
# """

import numpy as np
from src.quantylab.rltrader import utils

class Agent:
    # 에이전트 상태가 구성하는 값 개수(3가지 상태를 고려)
    # 주식 보유 비율, 손익률, 주당 매수 단가 대비 주가 등락률
    # 해당 부분을 추가하거나, 제거하여 상태 값 개수를 조절 가능!
    STATE_DIM = 3

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
    # TRADING_CHARGE = 0.00011  # 거래 수수료 0.011%
    # TRADING_CHARGE = 0  # 거래 수수료 미적용
    TRADING_TAX = 0.002  # 거래세 0.2%
    # TRADING_TAX = 0  # 거래세 미적용

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 관망
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(self, environment, initial_balance, min_trading_price, max_trading_price):
        # 현재 주식 가격을 가져오기 위해 환경 객체
        self.environment = environment
        self.initial_balance = initial_balance  # 초기 자본금

        # 최소 단일 매매 금액, 최대 단일 매매 금액
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        # Agent 클래스의 속성
        self.balance = initial_balance  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.portfolio_value = 0 # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}, 계좌 평단가
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 관망 횟수

        # Agent 클래스의 상태(STATE_DIM(3가지))
        self.ratio_hold = 0  # 주식 보유 비율
        self.profitloss = 0  # 현재 손익
        self.avg_buy_price = 0  # 주당 매수 단가

    def reset(self):
        """
        에이전트의 상태를 초기화 하는 함수
        한 에포크마다 호출
        """
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_buy_price = 0

    def set_balance(self, balance):
        """
        초기 자본금 설정
        """
        self.initial_balance = balance

    # 에이전트의 현재 상태를 반환
    def get_states(self):
        """
        에이전트 상태 반환
        retrun - tuple
        """

        # 주식 보유 비율 - (보유 주식 수 * 현재 주가) / 포트폴리오 가치  
        # 현재 상태에서 가장 많이 가질 수 있는 주식 수 대비 현재 보유한 주식의 비율
        self.ratio_hold = self.num_stocks * self.environment.get_price() \
            / self.portfolio_value

        # 24.04.29 - 가독성 있게 수정
        if self.avg_buy_price > 0:
            price_change_ratio = (self.environment.get_price() / self.avg_buy_price) - 1
        else:
            price_change_ratio = 0

        # 튜플로 반환
        return (
            self.ratio_hold,
            self.profitloss, # 손익률 = (포트폴리오 가치 / 초기 자본금) - 1
            price_change_ratio # 주식의 매수 가격 대비 주가 등락률 = (주가 / 주당 매수 단가) - 1
            # 보유 주식 수가 높을수록 현재 손익률과 주당 매수 단가 대비 등락률은 가까워진다.
            # (self.environment.get_price() / self.avg_buy_price) - 1 \
            #     if self.avg_buy_price > 0 else 0
        )

    # 에이전트가 취할 행동을 결정
    def decide_action(self, pred_value, pred_policy, epsilon):
        """
        탐험 또는 정책 신경망에 의한 행동 결정

        pred_value : 주어진 상태에서 각 행동의 예측된 가치
        pred_policy : 주어진 상태에서 각 행동을 선택할 확률을 나타내는 정책
        epsilon : 탐험, 에이전트가 무작위 행동을 선택할 확률을 결정
        return : action, confidence, exploration
                선택된 행동, 확신도, 탐험 여부를 반환
        """
        # 선택된 행동의 확신도
        confidence = 0.

        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

            # if pred_policy is not None:
            #     if np.max(pred_policy) - np.min(pred_policy) < 0.05:
            #         epsilon = 1

        # 탐험 결정
        # np.random.rand()가 epsilon보다 작으면 무작위 행동을 선택하고, 그렇지 않으면 예측된 값 중 가장 높은 행동을 선택
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred)

        # 선택된 행동에 대한 확신도를 계산 
        # pred_policy가 있으면 해당 행동의 확률을 사용하고, pred_value가 있으면 sigmoid 함수를 통해 확률로 변환된 값을 사용
        confidence = 0.5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        # 선택된 행동, 확신도, 탐험 여부를 반환
        return action, confidence, exploration

    def validate_action(self, action):
        """
        결정된 행동이 유효한지 판단, 매수 시 보유 현금이 충분한지, 매도 시 보유 주식이 있는지 확인
        """
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE):
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True

    
    def decide_trading_unit(self, confidence):
        """
        매수 또는 매도할 주식 수(거래 단위)를 결정 확신도(confidence)에 따라 달라지며, 
        최소 거래 가격과 최대 거래 가격 사이에서 결정
        정책 신경망이 결정한 행동의 신뢰가 높을수록 매수 또는 매도하는 단위를 크게함
        """
        if np.isnan(confidence): # confidence 값이 NaN (Not a Number)인 경우
            return self.min_trading_price # 확신도가 정의되지 않았을 때 최소 거래 단위로 거래를 진행하겠다는 의미
        added_trading_price = max(min(
            int(confidence * (self.max_trading_price - self.min_trading_price)),
            self.max_trading_price-self.min_trading_price), 0) # 맨 뒤에 영은 최소값 0 보장
        trading_price = self.min_trading_price + added_trading_price
        return max(int(trading_price / self.environment.get_price()), 1) # 현재 주가로 나눠 매수 or 매도 할 주식 수량 반환

    def act(self, action, confidence):
        """
        action : 탐험 또는 정책 신경망을 통해 결정된 행동으로 매수와 매도를 의미하는 0 또는 1의 값
        confidence : 정책 신경망을 통해 결정한 경우 결정한 행동에 대한 소프트 맥스 확률 값
        """

        # 매수, 매도가 가능한지 검증
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 주가 얻기
        curr_price = self.environment.get_price()

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 현재 현금 - (현재 주가 * (수수료) * 매수 단위
            balance = (
                self.balance - curr_price *
                (1 + self.TRADING_CHARGE) * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = min(
                    int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))), # 현재 현금 / 매수 주식의 주가(수수료 포함)
                    int(self.max_trading_price / curr_price) # 거래 정책에 따른 매수할 주식 수
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            if invest_amount > 0:
                self.avg_buy_price = \
                    (self.avg_buy_price * self.num_stocks + curr_price * trading_unit) \
                        / (self.num_stocks + trading_unit)  # 주당 매수 단가 갱신
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            # 보유 주식수를 최대 매도 단위로 설정
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            
            if invest_amount > 0:
                if self.num_stocks > trading_unit:
                    # 주식 매도 후의 새로운 평균 매수 단가 계산
                    self.avg_buy_price = (self.avg_buy_price * self.num_stocks - curr_price * trading_unit) \
                    / (self.num_stocks - trading_unit)
                else:
                    # 매도 후 보유 주식이 없을 경우 평균 매수 단가를 0으로 설정
                    self.avg_buy_price = 0
                    
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가

        # 관망
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 관망 횟수 증가

        # 포트폴리오 가치 갱신
        # PV = 주식 잔고 X 현재주가 + 현금잔고
        # PV(포트폴리오 가치)가 초기 자본금(initial_balance)보다 높으면 수익 발생, 적으면 손실
        self.portfolio_value = curr_price * self.num_stocks + self.balance
        self.profitloss = self.portfolio_value / self.initial_balance - 1
        return self.profitloss
