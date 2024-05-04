# """
# 신경망 모듈
# """

import threading
import abc
import numpy as np

import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Network:  
    # A3C 에서 필요, 스레드로 여러 A2C를 동시에 학습 진행해야 함
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, num_steps=1, lr=0.001, 
                shared_network=None, activation='sigmoid', loss='mse'):
        """
        매개변수 : 입력 데이터 크기, 출력 데이터 크기, 학습 속도, 공유 신경망, 활성화 함수, 학습 손실
        """
        self.input_dim = input_dim # 입력 데이터의 차원
        self.output_dim = output_dim # 출력 데이터의 차원
        self.num_steps = num_steps
        self.lr = lr # 신경망의 학습 속도
        self.shared_network = shared_network # 신경망의 상단부로 여러 신경망이 공유할 수 있음
        # 예를들어 A2C에서는 가치 신경망과 정책 신경망이 신경망의 상단부를 공유하고 하단 부분만 가치 예측과 확률 에측을 위해 달라짐
        self.activation = activation # 신경망의 출력 레이어 활성화 함수 
        self.loss = loss # 신경망의 손실 함수
        
        # CNN, LSTM : inp = (num_steps, input_dim)
        # DNN :  inp = (input_dim, )
        inp = None # 신경망 입력 데이터의 형태
        
        if self.num_steps > 1:
            inp = (self.num_steps, input_dim)
        else:
            inp = (self.input_dim,)

        # 공유 신경망 사용
        self.head = None
        if self.shared_network is None:
            self.head = self.get_network_head(inp, self.output_dim)
        else:
            self.head = self.shared_network
        
        # 공유 신경망 미사용
        # self.head = self.get_network_head(inp, self.output_dim)

        self.model = torch.nn.Sequential(self.head) # 신경망 모델 생성
        if self.activation == 'linear':
            pass
        elif self.activation == 'relu':
            self.model.add_module('activation', torch.nn.ReLU())
        elif self.activation == 'leaky_relu':
            self.model.add_module('activation', torch.nn.LeakyReLU())
        elif self.activation == 'sigmoid':
            self.model.add_module('activation', torch.nn.Sigmoid())
        elif self.activation == 'tanh':
            self.model.add_module('activation', torch.nn.Tanh())
        elif self.activation == 'softmax':
            self.model.add_module('activation', torch.nn.Softmax(dim=1))
        self.model.apply(Network.init_weights) # 신경망 가중치 초기화
        self.model.to(device)

        # RMSprop은 각 파라미터의 최근 그래디언트 값들의 제곱의 지수 이동 평균을 계산하고, 이 값을 사용하여 각 파라미터의 학습률을 조정 
        # 이 방식은 각 파라미터가 받는 업데이트 크기를 독립적으로 조정하여, 보다 안정적이고 빠른 수렴 가능
        # optimizer(옵티마이저)는 모델의 파라미터를 업데이트하여 학습 과정을 통해 손실 함수(loss function)의 값을 
        # 최소화하거나 최대화하는 데 사용되는 알고리즘 또는 방법론, RMSprop 외에도 Adam 같은 다른 것들이 있음   
        """
        주요 옵티마이저 종류
        SGD (Stochastic Gradient Descent):
        가장 기본적인 형태의 옵티마이저로, 각 반복에서 임의의 데이터 샘플(또는 작은 배치)을 사용하여 그래디언트를 계산하고 파라미터를 업데이트
        Momentum:
        SGD에 동적 관성을 추가하여 지역 최소값(local minima)에서 쉽게 벗어날 수 있도록 합니다. 이 방법은 이전 그래디언트 업데이트를 고려하여 현재 업데이트 방향을 조절
        Adagrad:
        각 파라미터에 대해 개별적인 학습률을 적용합니다. 많이 사용된 파라미터는 더 작은 학습률을, 적게 사용된 파라미터는 더 큰 학습률을 받음
        RMSprop:
        Adagrad의 확장으로, 학습률을 파라미터별로 적응적으로 조정하며, 과거 모든 그래디언트를 동등하게 고려하는 대신 가장 최근의 그래디언트에 더 큰 가중치를 부여
        Adam (Adaptive Moment Estimation):
        Momentum과 RMSprop의 아이디어를 결합한 옵티마이저로, 모멘텀과 스케일에 따른 학습률 조정을 동시에 수행
        """
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr) # lr = 학습률
        # self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.lr)

        # 손실 함수 정의
        self.criterion = None
        if loss == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif loss == 'binary_crossentropy':
            self.criterion = torch.nn.BCELoss()

    def predict(self, sample):
        """
        신경망을 통해 투자 행동별(매수, 매도, 관망) 예측값 반환
        가치 신경망의 경우 샘플에 대한 행동의 가치,
        정책 신경망의 경우 각 행동의 확률값  
        """
        with self.lock: # self.lock을 사용하여 스레드간의 간섭 방지
            self.model.eval() # self.model.eval()을 호출하여 모델을 평가(evaluation) 모드로 설정, 드롭아웃 등 비활성화
            # 예측에서는 기울기 계산이 필요없음
            with torch.no_grad(): # torch.no_grad() 컨텍스트 매니저를 사용하여 이 블록 내의 연산에서는 그래디언트 계산을 하지 않도록 설정
                x = torch.from_numpy(sample).float().to(device) # NumPy 배열에서 PyTorch 텐서로 샘플 데이터 변환
                pred = self.model(x).detach().cpu().numpy() # self.model(x)을 호출하여 입력 텐서 x에 대한 예측을 수행
                # detach()를 호출하여 계산 그래프에서 이 결과를 분리
                # cpu()를 사용하여 데이터를 CPU 메모리로 이동시킨 후, .numpy()를 호출하여 결과를 NumPy 배열로 변환
                pred = pred.flatten() # 다차원 > 1차원 배열로 평탄화
            return pred

    def train_on_batch(self, x, y):
        """
        배치 학습을 위한 데이터 생성
        """

        # self.num_steps가 1보다 큰 경우, x는 시퀀스 데이터를 포함하므로, 
        # 각 입력이 (self.num_steps, self.input_dim)의 형태를 갖도록 재구성
        if self.num_steps > 1:
            x = np.array(x).reshape((-1, self.num_steps, self.input_dim)) # x를 넘파이 배열로 변환 및 형태 재구성
            # -1을 사용하는 것은 해당 차원의 크기를 자동으로 추정
        else:
            x = np.array(x).reshape((-1, self.input_dim))

        loss = 0.

        with self.lock: # 스레드 락
            self.model.train() # 학습 모드로 설정
            _x = torch.from_numpy(x).float().to(device) # to(device)로 적절한 계산 장치(CPU 또는 GPU)에 할당
            _y = torch.from_numpy(y).float().to(device)
            y_pred = self.model(_x) # _x에 대한 예측
            _loss = self.criterion(y_pred, _y) # 예측값과, 레이블 _y 간의 손실 계산
            self.optimizer.zero_grad() # 기울기 초기화
            _loss.backward() # 손실 기울기 계산
            self.optimizer.step() # 기울기를 사용해 모델의 파라미터 업데이트
            loss += _loss.item() # numpy 값으로 손실을 변환하고 loss에 추가
        return loss

    def train_on_batch_for_ppo(self, x, y, a, eps, K):
        """
        Proximal Policy Optimization (PPO)
        정책의 업데이트가 너무 멀리 가지 않도록 제한을 두어, 학습의 안정성을 유지
        """
        if self.num_steps > 1:
            x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        else:
            x = np.array(x).reshape((-1, self.input_dim))
        loss = 0.
        with self.lock:
            self.model.train()
            _x = torch.from_numpy(x).float().to(device)
            _y = torch.from_numpy(y).float().to(device)
            probs = F.softmax(_y, dim=1)
            for _ in range(K):
                y_pred = self.model(_x)
                probs_pred = F.softmax(y_pred, dim=1)
                # 확률비(rto)를 계산하고, 이를 사용하여 조정된 확률비(rto_adv)와 제한된 확률비(clp_adv)를 계산
                rto = torch.exp(torch.log(probs[:, a]) - torch.log(probs_pred[:, a]))
                rto_adv = rto * _y[:, a]
                clp_adv = torch.clamp(rto, 1 - eps, 1 + eps) * _y[:, a]
                _loss = -torch.min(rto_adv, clp_adv).mean() # 두 값의 최소값을 손실 함수로 사용하여 평균 손실을 계산
                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
                loss += _loss.item()
        return loss

    @classmethod
    # @classmethod는 메서드가 클래스 메서드임을 나타내는 데코레이터
    # 클래스 메서드는 첫 번째 인자로 클래스 객체 cls를 자동으로 받음
    # 객체를 생성할 필요가 없음
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0, output_dim=0):
        """
        공유 신경망
        """
        if net == 'dnn':
            return DNN.get_network_head((input_dim,), output_dim)
        elif net == 'lstm':
            return LSTMNetwork.get_network_head((num_steps, input_dim), output_dim)
        elif net == 'cnn':
            return CNN.get_network_head((num_steps, input_dim), output_dim)

    @abc.abstractmethod
    # 추상 메서드 정의
    # 반드시 상속받는 하위 클래스에서 구현
    def get_network_head(inp, output_dim):
        pass

    @staticmethod
    def init_weights(m):
        """
        모델 m의 가중치 초기화
        """
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=0.01) # 정규 분포
        elif isinstance(m, torch.nn.LSTM):
            for weights in m.all_weights:
                for weight in weights:
                    torch.nn.init.normal_(weight, std=0.01) # 정규 분포로 반복적으로 초기화

    def save_model(self, model_path):
        """
        학습한 신경망을 파일로 저장
        """
        if model_path is not None and self.model is not None:
            torch.save(self.model, model_path)

    def load_model(self, model_path):
        """
        파일로 저장한 신경망을 파일로 저장
        """
        if model_path is not None:
            self.model = torch.load(model_path)
    
class DNN(Network):
    @staticmethod # 정적 메서드
    # @staticmethod 데코레이터는 메서드가 해당 클래스의 인스턴스나 클래스 객체 자체를 참조하지 않고 독립적으로 동작할 수 있음
    def get_network_head(inp, output_dim):
        """
        신경망의 상단부를 생성하는 클래스 함수
        배치 학습 : 큰 데이터 셋을 작은 데이터 셋으로 나누어 각각 학습 진행
        """
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]), # 배치 정규화
            torch.nn.Linear(inp[0], 256), # torch.nn.Linear(입력 특성(뉴런), 출력 특성(뉴런)), output=input⋅weight + bias
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )

    def predict(self, sample):
        """
        입력 데이터의 형태 조정
        """
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)


class LSTMNetwork(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_network_head(inp, output_dim):
        """
        2차원 입력 받음, num_steps 속성을 인자로 받음
        """
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            # inp[1]은 입력 데이터의 특성(feature)의 수
            # 128: LSTM 네트워크의 히든 유닛의 수
            # batch_first: 입력 데이터의 배치 크기가 첫 번째 차원에 오도록 지정
            # use_last_only: LSTM 네트워크의 출력에서 마지막 시퀀스의 결과만을 사용
            LSTMModule(inp[1], 128, batch_first=True, use_last_only=True), 
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )

    def predict(self, sample):
        sample = np.array(sample).reshape((-1, self.num_steps, self.input_dim))
        return super().predict(sample)


class LSTMModule(torch.nn.LSTM):
    def __init__(self, *args, use_last_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_last_only = use_last_only

    def forward(self, x):
        output, (h_n, _) = super().forward(x) # 입력 데이터 x를 순전파(입력층 > 출력층)처리하고 결과를 반환
        if self.use_last_only:
            return h_n[-1]
        return output # 각 시간 단계마다의 출력을 모두 포함하는 텐서 형식


class CNN(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_network_head(inp, output_dim):
        kernel_size = 2
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            torch.nn.Conv1d(inp[0], 1, kernel_size),
            torch.nn.BatchNorm1d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(inp[1] - (kernel_size - 1), 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        return super().predict(sample)
