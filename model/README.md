## 📊 Random Forest Regression 모델 활용


  - 다양한 변수와 복잡한 관계를 효과적으로 다룰 수 있음
  - 비선형적인 관계를 모델링하는 데 유리
  - 입력변수가 나이, 성별, 목적, 지역의 비선형적인 관계이므로 랜덤포레스트가 적합함


### 프로젝트 개요
랜덤 포레스트 회귀 모델을 사용하여 데이터를 학습하고, 예측값과 실제값 간의 관계를 분석합니다. 데이터 로드, 전처리, 학습, 평가, 시각화까지의 전체 과정을 다룹니다.

---

#### 📂 1. 필요한 라이브러리 임포트

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
```
- `numpy`: 수치 계산을 위한 라이브러리
- `pandas`: 데이터 프레임 처리
- `scikit-learn`: 모델 학습, 평가 및 데이터 분리
- `matplotlib`: 데이터 시각화

#### 📂 2. 데이터 로드 및 확인

```python
file_path = 'D:\\study\\class\\빅데\\random_forest_label_8_대피.csv'
rf_data = pd.read_csv(file_path)

print("Data Head:\n", rf_data.head())
print("Data Description:\n", rf_data.describe())
```
- 데이터를 지정된 경로에서 로드
- `head()`: 데이터의 상위 5개 행을 출력
- `scikit-learn`: 모델 학습, 평가 및 데이터 분리
- `describe()`: 데이터의 통계 요약 정보를 제공

#### 📂 3. 입력 변수와 출력 변수 분리
```python
X = rf_data[['gender', 'age', 'purpose', 'dest_hdong_cd']]
y = rf_data['score']

```
- `X`: 입력 변수
  - `gender`: 성별
  - `age`: 나이
  - `purpose`: 목적
  - `dest_hdong_cd`: 목적지 코드

- `y`: 출력 변수
  - `score`: 예측하려는 값
  
#### 📂 4. 데이터 분리
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- 데이터를 학습 데이터와 테스트 데이터로 나눔눔
- `test_size=0.2`: 테스트 데이터 비율은 20%로 설정
- `random_state=42`: 결과 재현성을 보장하기 위한 고정값

#### 📂 5. 랜덤 포레스트 모델 학습
```python
model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
```
- `n_estimators`: 랜덤 포레스트 트리의 개수 (100개)
- `max_depth`:각 트리의 최대 깊이 (20단계)
- `random_state`: 결과 재현성을 보장

#### 📂 6. 모델 예측
```python
y_pred = model.predict(X_test)
```
- 학습된 모델을 사용해 테스트 데이터의 값을 예측

#### 📂 7. 성능 평가
```python
mse = mean_squared_error(y_test, y_pred)max_depth=20, random_state=42)
r2 = r2_score(y_test, y_pred)
```
- 평가 결과
- `MSE (Mean Squared Error)`: 예측값과 실제값 간의 오차를 평가
- `R² (R-squared Score)`: 모델의 성능을 측정 (1에 가까울수록 좋음)

#### 📂 8. Feature Importance 확인
```python
importances = model.feature_importances_
```
- 각 입력 변수`(X)`가 모델 예측에 미치는 영향을 출력

### 모델 가중치

| Model | Acc     | Weight               |
| :-------- | :------- | :------------------------- |
| **의사 결정 트리**| `string` | **Required**. Your API key |