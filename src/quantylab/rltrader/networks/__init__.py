import os

# 환경 변수 RLTRADER_BACKEND 설정 필요, 없는 경우 pytorch 선택
if os.environ.get('RLTRADER_BACKEND', 'pytorch') == 'pytorch':
    print('Enabling PyTorch...')
    # 환경 변수가 pytorch 인 경우 모듈 불러옴
    from src.quantylab.rltrader.networks.networks_pytorch import Network, DNN, LSTMNetwork, CNN
else:
    print('Enabling TensorFlow...')
    from src.quantylab.rltrader.networks.networks_keras import Network, DNN, LSTMNetwork, CNN

# __init__.py 파일로, 해당 디렉토리 import * 하면, __all__ 것들 불러옴 
__all__ = [
    'Network', 'DNN', 'LSTMNetwork', 'CNN'
]
