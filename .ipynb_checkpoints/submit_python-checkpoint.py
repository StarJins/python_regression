from sklearn import datasets, linear_model, preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

data = pd.read_csv("./CALL_NDELIVERY_07MONTH.csv")
data = data.drop('시도', axis=1)

# 계절 추가
date = list(data.일자)
season = list()

for x in date:
    month = int(x % 10000 / 100)
    if month in [3, 4, 5]:
        season.append('봄')
    elif month in [6, 7 ,8]:
        season.append('여름')
    elif month in [6, 7 ,8]:
        season.append('가을')
    else:
        season.append('겨울')

data['계절'] = season

# 공휴일 추가
holiday_list = [20180101, 20180215, 20180216, 20180217, 20180301, 20180505, 20180522, 20180606, 20180815, 20180923, 20180924, 20180925, 20181003, 20181009, 20181225]

date = list(data.일자)
holiday = list()

for x in date:
    if x in holiday_list:
        holiday.append(1)
    else:
        holiday.append(0)

data['공휴일'] = holiday

# 일자 -> 월로 바꾸기
date = list(data.일자)
months = list()

for x in date:
    month = int(x % 10000 / 100)
    months.append(month)

data['월'] = months

# 업종 선택
data_c = data[data['업종'] == '치킨']

data_dummy = pd.get_dummies(data=data_c, columns=['요일'], drop_first=True)
data_dummy = pd.get_dummies(data=data_dummy, columns=['시간대'], drop_first=True)
data_dummy = pd.get_dummies(data=data_dummy, columns=['계절'], drop_first=True)
data_dummy = pd.get_dummies(data=data_dummy, columns=['월'], drop_first=True)
data_dummy = pd.get_dummies(data=data_dummy, columns=['시군구'], drop_first=True)
data_dummy = pd.get_dummies(data=data_dummy, columns=['읍면동'], drop_first=True)
data_dummy = data_dummy.drop('일자', axis=1)
data_dummy = data_dummy.drop('업종', axis=1)

# 인공 신경망
features = data_dummy.drop('통화건수', axis=1)
#print(features)

X = features.values
y = data_c['통화건수'].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

# data 정규화
scaler = StandardScaler()
scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


models = [
    linear_model.LinearRegression(), 
    MLPRegressor(hidden_layer_sizes=[512, 4], max_iter=5000, alpha=0.005, random_state=42), 
    RandomForestRegressor(n_estimators=100, random_state=0), 
    GradientBoostingRegressor(n_estimators=100, max_depth=10, criterion='mse')
]

with open('./model_scores.txt', 'w') as f:
    for m in models:
        m.fit(X_train, y_train)
        f.write(m.__class__ + '\n')
        f.write('Training Set Mean Squared Error: {:.2f}\n'.format(mean_squared_error(y_train, m.predict(X_train))))
        f.write('training Set R^2: {:.2f}\n'.format(r2_score(y_train, m.predict(X_train))))

        f.write('Training Set Mean Squared Error: {:.2f}\n'.format(mean_squared_error(y_test, m.predict(X_test))))
        f.write('testing R^2: {:.2f}\n\n'.format(r2_score(y_test, m.predict(X_test))))
