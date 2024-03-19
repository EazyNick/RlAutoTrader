## 환경설정
- [Anaconda 3.7+](https://www.anaconda.com/distribution/)
- [TensorFlow 2.7.0](https://www.tensorflow.org/)
  - `pip install tensorflow==2.7.0`
- [plaidML](https://plaidml.github.io/plaidml/)
  - `pip install plaidml-keras==0.7.0`
  - `pip install mplfinance`
- [PyTorch](https://pytorch.org/)

### Example Setup

```
conda create -n rltrader python=3.10.10
conda activate rltrader
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

# 개발 환경

- Python 3.6+
- PyTorch 1.10.1
- TensorFlow 2.7.0
- Keras 2.7.0 (TensorFlow에 포함되어 있음)

# conda 환경에서 TF 설치

## TF 1.15

> TF 1.15 사용을 위해서 Python 3.6을 설치한다.
> TF 1.15 사용할 경우 cuda 10.0, cudnn 7.4.2 (7.3.1) 설치해야 한다.
> https://www.tensorflow.org/install/source#tested_build_configurations
> https://github.com/tensorflow/models/issues/9706

```bash
conda create -n rltrader python=3.6
conda activate rltrader
pip install tensorflow-gpu==1.15
conda install cudatoolkit=10.0
conda install cudnn=7.3.1
pip install numpy
pip install pandas
```

## TF 2.5
https://www.tensorflow.org/install/source_windows?hl=en#gpu

```bash
conda create -n rltrader2 python=3.6
conda activate rltrader2
pip install tensorflow==2.5
```

CUDA 11.2
cuDNN 8.1

PATH
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp

## PyTorch

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

# 실행

- `main.py`를 통해 Command Line에서 RLTrader 실행 가능합니다. `run_e3.cmd`를 참고해 주세요.
- RLTrader 모듈들을 임포트하여 사용하는 것도 가능합니다. `main.py` 코드를 참고해 주세요.

## 예시

```bash
python main.py --mode train --ver v3 --name 005930 --stock_code 005930 --rl_method a2c --net dnn --start_date 20180101 --end_date 20191231
```

# 학습데이터

[퀀티랩 네이버 카페](https://cafe.naver.com/quantylab)에서 데이터 다운받을 수 있습니다. RLTrader 학습데이터 메뉴를 확인해 주세요.

학습데이터는 v1, v2, v3, v4 버전이 존재합니다. 주로 v3, v4를 사용하시면 됩니다. v3는 종목 데이터에 일부 중요한 시장 데이터를 추가한 학습데이터 입니다. v4는 v3에 시장데이터를 대량 추가한 데이터 입니다.

v3, v4 학습데이터는 시장 데이터와 종목 데이터를 합하여 사용하면 됩니다. `data_manager.py`를 참고해 주세요.

## v3

- 종목 데이터
  - `date,open,high,low,close,volume,per,pbr,roe,open_lastclose_ratio,high_close_ratio,low_close_ratio,diffratio,volume_lastvolume_ratio,close_ma5_ratio,volume_ma5_ratio,close_ma10_ratio,volume_ma10_ratio,close_ma20_ratio,volume_ma20_ratio,close_ma60_ratio,volume_ma60_ratio,close_ma120_ratio,volume_ma120_ratio,ind,ind_diff,ind_ma5,ind_ma10,ind_ma20,ind_ma60,ind_ma120,inst,inst_diff,inst_ma5,inst_ma10,inst_ma20,inst_ma60,inst_ma120,foreign,foreign_diff,foreign_ma5,foreign_ma10,foreign_ma20,foreign_ma60,foreign_ma120`
- 시장 데이터
  - `date, market_kospi_ma5_ratio,market_kospi_ma20_ratio,market_kospi_ma60_ratio,market_kospi_ma120_ratio,bond_k3y_ma5_ratio,bond_k3y_ma20_ratio,bond_k3y_ma60_ratio,bond_k3y_ma120_ratio`

## v4

- 종목 데이터: v3 종목 데이터와 동일
- 시장 데이터: v3 시장 데이터에 다음 시장 데이터 추가
  - TBD

# 프로파일링
- `python -m cProfile -o profile.pstats main.py ...`
- `python profile.py`

# Tips

- Windows Power Shell에서 로그 Tail 하는 방법

```
cat D:\dev\rltrader\output\train_000240_a2c_lstm\train_000240_a2c_lstm.log -Wait -Tail 100
```

# Troubleshooting

## TF 1.15에서 다음 에러가 나면 Python 3.6으로 맞춰준다.
```
NotImplementedError: Cannot convert a symbolic Tensor (lstm/strided_slice:0) to a numpy array.
```

## original_keras_version = f.attrs['keras_version'].decode('utf8') AttributeError: 'str' object has no attribute 'decode'
```
https://github.com/keras-team/keras/issues/14265
https://pypi.org/project/h5py/#history
pip install h5py==2.10.0
```
