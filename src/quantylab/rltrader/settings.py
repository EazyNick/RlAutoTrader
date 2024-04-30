import os
import locale
import platform

# 로거 이름
LOGGER_NAME = 'rltrader'

# 경로 설정
# 환경 변수 'RLTRADER_BASE', 설정 안되어 있으면 두번째 인자를 사용
# os.path.abspath(...): 주어진 경로를 절대 경로로 변환
# 현재 실행 중인 파일(__file__)의 경로를 기준으로 주어진 상대경로를 조합하여 새로운 경로를 생성
# os.path.pardir: 상위 디렉토리를 나타내는 문자열
# 현재 파일로부터 네 단계 위의 디렉토리를 절대 경로로 나타냄
BASE_DIR = os.environ.get('RLTRADER_BASE', 
    os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

# 로케일 설정
if 'Linux' in platform.system() or 'Darwin' in platform.system():
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
elif 'Windows' in platform.system():
    locale.setlocale(locale.LC_ALL, '')
