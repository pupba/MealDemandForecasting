# 식사 인원 예측 머신러닝 모델 (The Development of Prediction Models for the Number of People for Meal at University Cafeteria)

## 라이선스 (License)

⭐️ 이 프로젝트 및 리포지토리는 [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) 하에 배포됩니다. 이 라이선스에 따라 상업적인 목적으로 사용, 판매, 또는 배포하는 것이 금지됩니다. 비영리 목적으로만 사용이 허용됩니다. 자세한 정보는 링크를 클릭하여 확인하세요.

## 개요 (Overview)

⭐️ 이 리포지토리는 대학교 교내식당을 위한 식사 인원 예측 모델을 개발하는 프로젝트입니다.

⭐️ 이 모델은 실제 식당 데이터를 사용해 식사 인원 예측 모델을 개발하여 목포해양대학교 교내식당에서 발생하 는 음식 품절과 대량의 잔반 발생을 줄이고자 한다.

-   **프로젝트 목표**: 대학교 교내식당의 식사 인원을 예측하는 머신러닝 모델 개발.
-   **데이터셋**: 목포해양대학교 식당 데이터(메뉴, 식사인원 등), 기상청 데이터(기온, 습도, 강수량)
-   **모델**: 선형 회귀, 딥러닝 등 다양한 머신러닝 알고리즘 비교 및 최적 모델 선택.
-   **결과**: 예측 모델의 정확도 평가 및 개선.

## 예측 프로세스

![식사 인원 예측 프로세스](https://github.com/pupba/MealDemandForecasting/assets/53106728/bcb02705-6c89-4b66-a8ee-ad0a9f7bb9a6)

## 사용된 기술 (Technologies Used)

#### Python 3.11.0

#### Python Library

-   Lasso
-   Ridge
-   Elastic Net
-   XGBoostRegressor
-   SVM
-   Pandas
-   Numpy
-   Matplotlib
-   Seaborn

## 프로젝트 구성 (Project Structure)

프로젝트 디렉토리 구조는 다음과 같습니다:

-   `Data/`: 전처리 결과 데이터(리포지토리에는 제외)
-   `OriginData/`: 데이터셋(리포지토리에는 제외)
-   `modules/`: 전처리를 위한 모듈들 저장
-   `README.md`: 프로젝트 소개 및 사용법 문서.
-   `LICENSE`: 프로젝트 라이선스 정보.

## 데이터 수집 틀

![image](https://github.com/pupba/controlsystem/assets/53106728/156ee3ce-d4e0-48b0-8b3c-090172b1c461)

## 시각화 결과 예시

![2018년 월별 총 식사 인원](https://github.com/pupba/MealDemandForecasting/assets/53106728/db20c6dd-b96f-4704-8378-480ac5598ed6)
![2018년 요일별 총 식사 인원](https://github.com/pupba/MealDemandForecasting/assets/53106728/3bce3178-0e12-477a-8c01-1cf74a562d15)
![2018년 강우 종류별 식사 인원](https://github.com/pupba/MealDemandForecasting/assets/53106728/1bb4fee5-036c-4fee-b583-1250d87794d4)
![2018년 평균기온 별 총 식사 인원](https://github.com/pupba/MealDemandForecasting/assets/53106728/c7c5a8ec-cb21-483b-b93a-c56692f4c347)
![2018년 석식 메뉴 Top5 빈도수](https://github.com/pupba/MealDemandForecasting/assets/53106728/0d30b102-6cfe-4257-a24a-f38362be8ed0)
![2018년 TOP5 메뉴와 식사 인원](https://github.com/pupba/MealDemandForecasting/assets/53106728/815d3050-7d15-41d0-a856-a883f1995b2f)
![2018년 불쾌한 상태 유무에 대한 식사인원 분포](https://github.com/pupba/MealDemandForecasting/assets/53106728/d2c68cfb-4f40-47c1-8889-8a26ca8f567e)
![2018년, 중식 데이터 컬럼간의 상관관계 분석
![2018년 평균습도 별 총 식사 인원](https://github.com/pupba/MealDemandForecasting/assets/53106728/3b1e13cb-10dd-4acd-9c91-325941f1d951)
](https://github.com/pupba/MealDemandForecasting/assets/53106728/c2619801-7340-4e6c-b73f-f34e81409c2d)
![2018년 이벤트 종류에 따른 총 식사 인원](https://github.com/pupba/MealDemandForecasting/assets/53106728/a94a5f98-c832-4569-a916-700e3c3e2a11)

## Learning Curve

### Lasso

![lasso_m](https://github.com/pupba/MealDemandForecasting/assets/53106728/ef16e2a6-49f6-4217-a0de-6c7a77842c36)
![lasso_l](https://github.com/pupba/MealDemandForecasting/assets/53106728/d7adaab5-a47f-419c-935a-1ae481f056e8)
![lasso_d](https://github.com/pupba/MealDemandForecasting/assets/53106728/a327a5df-736f-40e5-a172-fe2bc41e0183)

### Ridge

![ridge_m](https://github.com/pupba/MealDemandForecasting/assets/53106728/73a163c0-9872-4d99-b086-bc3586a0bee3)
![ridge_l](https://github.com/pupba/MealDemandForecasting/assets/53106728/16d90d51-aad3-4bcf-be06-ce3d0d2c73aa)
d4c3ed45-7363-4adf-8f5e-28a01b80bd0a)
![ridge_d](https://github.com/pupba/MealDemandForecasting/assets/53106728/2c9c018c-e1dc-4123-9cf3-56d54749c066)

### ElasticNet

![EN_m](https://github.com/pupba/MealDemandForecasting/assets/53106728/65026d8e-9b45-46f7-bf72-a9c7edcdc128)
![EN_l](https://github.com/pupba/MealDemandForecasting/assets/53106728/c9468c2f-1da8-4605-bf4e-97d780e7c427)
![EN_d](https://github.com/pupba/MealDemandForecasting/assets/53106728/4e766015-8202-45a4-8b31-fb88e39ac769)

### SVM

![SVM_m](https://github.com/pupba/MealDemandForecasting/assets/53106728/faf9c52e-730e-42fe-ae91-fbc39cae65e1)
![SVM_l](https://github.com/pupba/MealDemandForecasting/assets/53106728/a274ef70-7ab8-44b7-8630-42317f4cd6e6)
![SVM_d](https://github.com/pupba/MealDemandForecasting/assets/53106728/8ec

### XGBoostRegressor

![XGBR_m](https://github.com/pupba/MealDemandForecasting/assets/53106728/b9bb3a2b-e7b7-4889-aa2b-587530a04b1a)
![XGBR_l](https://github.com/pupba/MealDemandForecasting/assets/53106728/d1682956-f5e5-44ae-b9a4-e69eb56e891e)
![XGBR_d](https://github.com/pupba/MealDemandForecasting/assets/53106728/4b1dd6d9-46e1-408e-9255-87eb1447bf13)

## Evaluation of Prediction Models(MAE)

| 모델         | MAE  |
| ------------ | ---- |
| SVM (조식)   | 44명 |
| Lasso (조식) | 47명 |
| Ridge (조식) | 47명 |
| XGBR (조식)  | 42명 |
| EN (조식)    | 47명 |
| SVM (중식)   | 48명 |
| Lasso (중식) | 47명 |
| Ridge (중식) | 48명 |
| XGBR (중식)  | 58명 |
| EN (중식)    | 47명 |
| XGBR (석식)  | 12명 |
| EN (석식)    | 11명 |
| Lasso (석식) | 11명 |
| Ridge (석식) | 11명 |
| SVM (석식)   | 11명 |

## Conclusions

-   연구 초기 단계이고 충분한 데이터 표본이 모이지 않아 모델이 예측을 엄청 잘하지는 못한다. 추후에 데이터 점점 쌓이게 되면 더 좋은 성능의 예측 모델을 만들어 낼 수 있을 것 같다.
