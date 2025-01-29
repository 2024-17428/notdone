import requests
import pandas as pd
import time
import numpy as np
from bs4 import BeautifulSoup
from textblob import TextBlob
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyupbit
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
access_key = os.getenv("UPBIT_ACCESS_KEY")
secret_key = os.getenv("UPBIT_SECRET_KEY")
upbit = pyupbit.Upbit(access_key, secret_key)

# 2. 학습된 강화학습 모델 로드
model = load_model("dqn_bitcoin_trading_episode_27.keras")

# 3. 뉴스 감정 분석 함수
def get_latest_news(keyword="비트코인"):
    """
    네이버 뉴스에서 최신 비트코인 뉴스를 크롤링하여 반환합니다.
    """
    base_url = "https://search.naver.com/search.naver?where=news"
    params = {"query": keyword, "sm": "tab_pge", "sort": "0", "start": 1}
    response = requests.get(base_url, params=params)
    soup = BeautifulSoup(response.text, "html.parser")
    news_items = soup.find_all("a", class_="news_tit")
    news_data = []
    for item in news_items:
        title = item.get("title")
        link = item.get("href")
        news_data.append({"title": title, "link": link})
    return news_data

def sentiment_analysis(news_title):
    """
    뉴스 제목에 대해 감정 분석을 실행하고 점수를 반환합니다.
    """
    try:
        translated = TextBlob(news_title)
        sentiment = translated.sentiment.polarity  # -1(부정) ~ 1(긍정)
    except Exception as e:
        print(f"감정 분석 오류: {e}")
        sentiment = 0
    return sentiment

# 4. 상태 관찰 함수 (감정 점수 제외하고 3개 변수로만 입력)
previous_sentiment = 0  # 이전 감정 점수를 저장하는 변수

def get_current_state():
    """
    현재 업비트 1분봉 데이터를 가져오고 상태 벡터를 반환합니다.
    """
    global previous_sentiment

    ticker = "KRW-BTC"
    df = pyupbit.get_ohlcv(ticker, interval="minute1", count=1)
    latest = df.iloc[-1]
    
    # 최신 뉴스 데이터 가져오기
    latest_news = get_latest_news()
    if latest_news:
        sentiment_score = sentiment_analysis(latest_news[0]["title"])
        previous_sentiment = sentiment_score  # 최신 감정 점수로 갱신
    else:
        sentiment_score = previous_sentiment  # 뉴스가 없으면 이전 감정 점수 유지
    
    # 3개의 값만 상태 벡터에 사용
    state = [
        latest["close"],  # 종가
        latest["volume"],  # 거래량
        sentiment_score  # 감정 점수 (기존 코드에서 변경하지 않음, 향후 다른 데이터로 대체 가능)
    ]
    return np.array(state, dtype=np.float32).reshape(1, -1)
def execute_trade(action, ticker="KRW-BTC", buy_price=None):
    """
    강화학습 모델의 행동에 따라 매수, 매도, 대기를 실행합니다.
    매수 후 매도 조건: 상승 2%, 하락 -2%일 때 매도
    """
    if action == 1:  # 매수
        krw_balance = upbit.get_balance(ticker="KRW")  # 원화 잔액 확인
        print(krw_balance)
        if krw_balance > 5000:  # 최소 주문 금액 제한
            upbit.buy_market_order(ticker, krw_balance * 0.995)  # 원화 잔액의 99.5% 매수
            print(f"매수 실행 (잔액: {krw_balance} KRW)")
            buy_price = pyupbit.get_orderbook(ticker)["orderbook_units"][0]["ask_price"]  # 매수 가격 저장
            print(f"매수 가격: {buy_price}")
    elif action == 2:  # 매도
        if buy_price is not None:  # buy_price가 None이 아닌지 확인
            btc_balance = upbit.get_balance("BTC")  # 비트코인 잔량 확인
            if btc_balance > 0:  # 최소 거래량 제한
                current_price = pyupbit.get_orderbook(ticker)["orderbook_units"][0]["bid_price"]  # 현재 비트코인 가격
                # 매수 가격 대비 +2% 또는 -2%일 때 매도
                if current_price >= buy_price * 1.02 or current_price <= buy_price * 0.98:
                    upbit.sell_market_order(ticker, btc_balance)  # 보유 BTC 전량 매도
                    print(f"매도 실행 (보유량: {btc_balance} BTC, 가격: {current_price})")
                else:
                    print(f"매도 조건 미달 (현재 가격: {current_price}, 매수 가격: {buy_price})")
        else:
            print("매도할 때 buy_price가 없습니다. 매수 후 매도를 진행해야 합니다.")
    else:
        print("대기 중...")

    return buy_price


# 6. 자동매매 루프
def trade_loop():
    """
    1분 단위로 상태를 관찰하고 강화학습 모델의 행동을 기반으로 자동매매를 수행합니다.
    """
    buy_price = None  # 매수 가격 초기화
    try:
        while True:
            # 현재 상태 가져오기
            state = get_current_state()

            # 강화학습 모델로 행동 예측
            action = np.argmax(model.predict(state, verbose=0)[0])  # 0: 대기, 1: 매수, 2: 매도
            print(f"예측된 행동: {action}")

            # 매매 실행
            buy_price = execute_trade(action, buy_price=buy_price)

            # 1분 대기
            time.sleep(60)
    except Exception as e:
        print(f"오류 발생: {e}")

# 실행
trade_loop()
