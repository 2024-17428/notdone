import requests
import pandas as pd
import time
import gym
import numpy as np
from bs4 import BeautifulSoup
from textblob import TextBlob
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random

# 1. 업비트 데이터 가져오기 (3개월치 분봉 데이터)
def get_full_upbit_data(market="KRW-BTC", interval="minutes/1", count=200, days=30):
    url = f"https://api.upbit.com/v1/candles/{interval}"
    all_data = []
    to = None
    
    for _ in range(days * 24 * 60 // (count - 1)):  # 필요한 요청 횟수 계산
        params = {"market": market, "count": count}
        if to:
            params["to"] = to
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print("API 요청 오류:", response.status_code, response.text)
            break
        
        data = response.json()
        all_data.extend(data)
        to = data[-1]["candle_date_time_utc"]  # 마지막 데이터 기준 시간 업데이트
        
        time.sleep(0.1)  # 요청 제한 방지
    
    df = pd.DataFrame(all_data)
    df = df[["candle_date_time_kst", "opening_price", "high_price", "low_price", "trade_price", "candle_acc_trade_volume"]]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df

# 3개월치 데이터 가져오기
btc_data = get_full_upbit_data(days=30)
print(f"가져온 데이터 개수: {len(btc_data)}")
print(btc_data.head())

# 2. 네이버 뉴스 데이터 가져오기 (3개월치)
def get_news_data(keyword="비트코인", days=30):
    base_url = "https://search.naver.com/search.naver?where=news"
    news_data = []
    today = pd.Timestamp.now()

    for day_offset in range(days):
        date = (today - pd.Timedelta(days=day_offset)).strftime("%Y%m%d")
        params = {
            "query": keyword,
            "sm": "tab_pge",
            "sort": "0",
            "ds": date,
            "de": date,
            "start": 1
        }
        response = requests.get(base_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser")
        
        news_items = soup.find_all("a", class_="news_tit")
        for item in news_items:
            title = item.get("title")
            news_data.append({"date": date, "title": title})
        
    return pd.DataFrame(news_data)

# 비트코인 뉴스 데이터 가져오기
news_data = get_news_data()
print(news_data.head())

# 3. 뉴스 감정 분석 및 점수 유지
def sentiment_analysis(news_title):
    translated = TextBlob(news_title)
    sentiment = translated.sentiment.polarity  # -1(부정) ~ 1(긍정)
    return sentiment

def prepare_sentiment_data(news_data, btc_data):
    """
    뉴스 데이터를 감정 분석한 후, 비트코인 데이터 길이에 맞춰 감정 점수를 채웁니다.
    """
    news_data["sentiment"] = news_data["title"].apply(sentiment_analysis)
    
    # 비트코인 데이터의 타임스탬프를 기준으로 빈 데이터프레임 생성
    sentiment_series = pd.DataFrame({"timestamp": btc_data["timestamp"], "sentiment": np.nan})
    
    # 뉴스 데이터의 타임스탬프와 감정 점수를 매핑
    for _, row in news_data.iterrows():
        timestamp = row["date"]
        sentiment_score = row["sentiment"]
        sentiment_series.loc[sentiment_series["timestamp"] >= pd.Timestamp(timestamp), "sentiment"] = sentiment_score
    
    # 감정 점수를 다음 뉴스 업데이트 전까지 유지 (결측값 보완)
    sentiment_series["sentiment"].fillna(method="ffill", inplace=True)
    sentiment_series["sentiment"].fillna(0, inplace=True)  # 초기값이 없을 경우 0으로 채움

    return sentiment_series

sentiment_series = prepare_sentiment_data(news_data, btc_data)
# 4. 강화학습 환경 정의# 4. 강화학습 환경 정의
class BitcoinTradingEnvWithSentiment(gym.Env):
    def __init__(self, data, sentiment_series, initial_balance=10000):
        super(BitcoinTradingEnvWithSentiment, self).__init__()
        self.data = data
        self.sentiment_series = sentiment_series
        self.initial_balance = initial_balance
        self.action_space = gym.spaces.Discrete(3)  # 0: 대기, 1: 매수, 2: 매도
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32  # 종가, 볼륨, 감정 점수
        )
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0  # 보유량
        self.current_step = 0
        self.done = False
        return self._get_observation()
    
    def _get_observation(self):
        obs = [
            self.data.iloc[self.current_step]["close"],  # 종가
            self.data.iloc[self.current_step]["volume"], # 거래량
            self.sentiment_series.iloc[self.current_step]["sentiment"]  # 감정 점수
        ]
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]["close"]
        reward = 0

        if action == 1:  # 매수
            # 수수료 반영
            transaction_value = self.balance / current_price  # 매수 금액
            transaction_fee = transaction_value * 0.0005  # 0.25% 수수료
            self.position += (self.balance - transaction_fee) / current_price  # 수수료 제외한 금액으로 매수
            self.balance = 0
            self.buy_price = current_price  # 매수 가격 저장
              # 빈번한 거래에 패널티 부여

        elif action == 2:  # 매도
            if self.position > 0:  # 비트코인 잔량이 있을 때만 매도
                # 가격 상승 1% 또는 하락 -1% 조건에 맞을 경우 매도
                if current_price >= self.buy_price * 1.02:  # 1% 상승
                    # 수수료 반영
                    transaction_value = self.position * current_price  # 매도 금액
                    transaction_fee = transaction_value * 0.0005  # 0.25% 수수료
                    self.balance += (transaction_value - transaction_fee)  # 수수료 제외한 금액
                    reward = self.balance - self.initial_balance  # 수수료를 고려한 보상
                    self.position = 0
                    reward += 0.1 
                      # 빈번한 거래에 패널티 부여
                elif current_price <= self.buy_price * 0.98:  # 손절 -1% 하락
                    # 수수료 반영
                    transaction_value = self.position * current_price  # 매도 금액
                    transaction_fee = transaction_value * 0.0025  # 0.25% 수수료
                    self.balance += (transaction_value - transaction_fee)  # 수수료 제외한 금액
                    reward = self.balance - self.initial_balance  # 수수료를 고려한 보상
                    self.position = 0
                      # 빈번한 거래에 패널티 부여
             # 매도할 수 없을 때 패널티

      
        total_asset = self.balance + self.position * current_price
        reward += total_asset - self.initial_balance  # 수수료 포함 후 이익 계산

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        return self._get_observation(), reward, self.done, {}

# 5. DQNAgent 정의
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 무작위 행동
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 6. 환경 및 에이전트 설정
env = BitcoinTradingEnvWithSentiment(btc_data, sentiment_series)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# 7. 학습 및 모델 저장
# 7. 학습 및 모델 저장
episodes = 100
batch_size = 32

# 첫 번째 에피소드부터 모델을 학습하고, 계속 업데이트하면서 저장
for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    final_asset = 0

    for step in range(len(btc_data) - 1):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            final_asset = env.balance + env.position * btc_data.iloc[env.current_step]["close"]
            break

    # 에피소드 결과 출력
    print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}, Final Asset: {final_asset}")
    
    # 모델 학습 (중첩 학습)
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # 매 에피소드마다 모델을 저장 (덮어쓰기)
    agent.model.save(f"dqn_bitcoin_trading_episode_{episode+1}.keras")
    print(f"Episode {episode+1} 모델이 저장되었습니다.")

# 마지막 모델 저장 (최종 모델)
agent.model.save("dqn_bitcoin_trading_final.keras")
print("최종 모델이 저장되었습니다.")