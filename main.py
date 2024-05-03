# """
# 실행 모듈
# """

import os
import sys
import logging
import argparse
import json

from src.quantylab.rltrader import settings
from src.quantylab.rltrader import utils
from src.quantylab.rltrader import data_manager


if __name__ == '__main__':
    # 동작 모드(--mode), 버전(--ver), 실행 이름(--name), 스톡 코드(--stock_code), 강화 학습 방법(--rl_method), 신경망 아키텍처(--net), 
    # 계산 백엔드(--backend), 데이터의 시작일과 종료일, 학습률(--lr), RL 알고리즘의 할인율, 거래 시뮬레이션의 시작 잔액을 설정하는 옵션을 포함
    parser = argparse.ArgumentParser() # 스크립트의 명령줄 인터페이스(CLI)를 정의
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default='train')
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3', 'v4', 'v4.1', 'v4.2'], default='v3')

    # 사용자가 실행 또는 작업의 이름을 지정할 수 있도록 하는 인수 --name을 추가합니다. 
    # 사용자가 이름을 제공하지 않으면 utils.get_time_str()을 사용하여 기본값이 생성되며, 이 값은 고유성을 보장하기 위해 타임스탬프가 있는 문자열을 만듭니다.
    parser.add_argument('--name', default='005930') 
    parser.add_argument('--stock_code', nargs='+', default='005930') # 종목코드
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'ppo', 'monkey'], default='dqn') # 알고리즘 선택
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default='lstm') #신경망 선택
    parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'plaidml'], default='pytorch') #신경망 연산을 위한 백엔드
    parser.add_argument('--start_date', default='20180101') #사용할(학습할) 데이터 시작일
    parser.add_argument('--end_date', default='20181231') #데이터 종료일

    # 학습률(learning rate)은 기계학습에서 모델이 학습하는 속도를 조절하는 매개변수로, 
    # 너무 높으면 학습이 불안정해질 수 있고 너무 낮으면 학습이 너무 느려질 수 있습니다.
    parser.add_argument('--lr', type=float, default=0.001)
    # 미래의 보상을 현재 가치로 환산할 때 사용하는 인자
    # 이 값이 낮으면 에이전트가 단기 보상에 더 큰 가치를 두게 되고, 높으면 장기 보상을 중시하게 됩니다.
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--balance', type=int, default=100000000) # 초기 자본금
    args = parser.parse_args()

    # 학습기 파라미터 설정
    # 구문 분석된 인수를 기반으로 변수를 초기화하고 출력 파일 및 엡실론 및 에포크 수와 같은 학습 매개 변수의 이름을 설정
    output_name = f'{args.mode}_{args.name}_{args.rl_method}_{args.net}'
    learning = args.mode in ['train', 'update']
    reuse_models = args.mode in ['test', 'update', 'predict']
    value_network_name = f'{args.name}_{args.rl_method}_{args.net}_value.mdl'
    policy_network_name = f'{args.name}_{args.rl_method}_{args.net}_policy.mdl'
    start_epsilon = 1 if args.mode in ['train', 'update'] else 0
    num_epoches = 1000 if args.mode in ['train', 'update'] else 1
    num_steps = 5 if args.net in ['lstm', 'cnn'] else 1

    # Backend 설정
    # 사용자의 선택(TensorFlow, PyTorch 또는 PlaidML)을 기반으로 신경망 계산을 위한 백엔드를 구성하여 필요한 환경 변수를 설정
    os.environ['RLTRADER_BACKEND'] = args.backend
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 생성
    # 설정 모듈의 기본 디렉터리를 사용하여 실행에 대한 출력 디렉터리를 만듭니다.
    output_path = os.path.join(settings.BASE_DIR, 'output', output_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    # 구문 분석된 명령줄 인수를 나중에 참조할 수 있도록 출력 디렉토리에 있는 params.json이라는 이름의 JSON 파일에 덤프
    params = json.dumps(vars(args))
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(params)

    # 모델 경로 준비
    # 모델 포멧은 TensorFlow는 h5, PyTorch는 pickle
    # 백엔드에 따라 다른 파일 형식을 가정하여 값 및 정책 네트워크를 저장하기 위한 파일 경로를 준비
    value_network_path = os.path.join(settings.BASE_DIR, 'models', value_network_name)
    policy_network_path = os.path.join(settings.BASE_DIR, 'models', policy_network_name)

    # 로그 기록 설정
    # 특정 형식과 두 개의 핸들러를 사용하여 로깅을 설정합니다.
    # 하나는 정보 수준의 stdout이고 다른 하나는 debug 수준의 파일을 사용하여 이전 로그 파일이 제거되도록 합니다.
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(settings.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(params)
    
    # Backend 설정, 로그 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from src.quantylab.rltrader.learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner, PPOLearner

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []

    # 차트 데이터, 학습 데이터 준비
    chart_data, training_data = data_manager.load_data(
        args.stock_code, args.start_date, args.end_date, ver=args.ver)

    assert len(chart_data) >= num_steps
    
    # 최소/최대 단일 매매 금액 설정
    min_trading_price = 100000
    max_trading_price = 10000000

    # 공통 파라미터 설정
    common_params = {'rl_method': args.rl_method, 
        'net': args.net, 'num_steps': num_steps, 'lr': args.lr,
        'balance': args.balance, 'num_epoches': num_epoches, 
        'discount_factor': args.discount_factor, 'start_epsilon': start_epsilon,
        'output_path': output_path, 'reuse_models': reuse_models}

    # 강화학습 시작
    learner = None
    if args.rl_method != 'a3c':
        common_params.update({'stock_code': args.stock_code,
            'chart_data': chart_data, 
            'training_data': training_data,
            'min_trading_price': min_trading_price, 
            'max_trading_price': max_trading_price})
        if args.rl_method == 'dqn':
            learner = DQNLearner(**{**common_params, 
                'value_network_path': value_network_path})
        elif args.rl_method == 'pg':
            learner = PolicyGradientLearner(**{**common_params, 
                'policy_network_path': policy_network_path})
        elif args.rl_method == 'ac':
            learner = ActorCriticLearner(**{**common_params, 
                'value_network_path': value_network_path, 
                'policy_network_path': policy_network_path})
        elif args.rl_method == 'a2c':
            learner = A2CLearner(**{**common_params, 
                'value_network_path': value_network_path, 
                'policy_network_path': policy_network_path})
        elif args.rl_method == 'ppo':
            learner = PPOLearner(**{**common_params, 
                'value_network_path': value_network_path, 
                'policy_network_path': policy_network_path})
        elif args.rl_method == 'monkey':
            common_params['net'] = args.rl_method
            common_params['num_epoches'] = 10
            common_params['start_epsilon'] = 1
            learning = False
            learner = ReinforcementLearner(**common_params)
    else:
        list_stock_code.append(args.stock_code)
        list_chart_data.append(chart_data)
        list_training_data.append(training_data)
        list_min_trading_price.append(min_trading_price)
        list_max_trading_price.append(max_trading_price)

    if args.rl_method == 'a3c':
        learner = A3CLearner(**{
            **common_params, 
            'list_stock_code': list_stock_code, 
            'list_chart_data': list_chart_data, 
            'list_training_data': list_training_data,
            'list_min_trading_price': list_min_trading_price, 
            'list_max_trading_price': list_max_trading_price,
            'value_network_path': value_network_path, 
            'policy_network_path': policy_network_path})
    
    assert learner is not None

    if args.mode in ['train', 'test', 'update']:
        learner.run(learning=learning)
        if args.mode in ['train', 'update']:
            learner.save_models()
    elif args.mode == 'predict':
        learner.predict()
