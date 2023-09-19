import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler
# 머신러닝 : Ridge, Lasso, ElasticNet, SVM, XGBoost
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
# 평가 지표
from sklearn.metrics import mean_absolute_error


class MachineLearningMate:
    def __init__(self) -> None:
        # 스케일러 객체
        self.scaler = StandardScaler()

    def __loadData(self) -> pd.DataFrame:
        PATH = "./Data/CleanedData"
        # 라벨링 데이터
        labels = [pd.read_excel(PATH+f"/{n}", engine="openpyxl")
                  for n in sorted(os.listdir(PATH), key=lambda x:int(x.split('_')[1])) if "label" in n]
        dataL = pd.concat(labels, axis=0)
        return dataL

    def __featureSelector(self) -> pd.DataFrame:
        dataL = self.__loadData()
        # Labeling
        label = {"Xm": None, "Ym": None, "Xl": None,
                 "Yl": None, "Xd": None, "Yd": None}
        # 독립 변수 - Feature
        for rg in [("m", 5, 9), ("l", 10, 14), ("d", 14, 19)]:
            label[f"X{rg[0]}"] = dataL.iloc[:, [1, 2, 3] +
                                            list(range(rg[1], rg[2]))+list(range(19, 28))]

        # 종속 변수 - Target
        for tg in [("m", "Breakfast"), ("l", "Lunch"), ("d", "Dinner")]:
            label[f"Y{tg[0]}"] = dataL.loc[:, f"{tg[1]} Attendance"]

        return label

    def __splitTrainTestData(self, data: dict) -> list:
        result = []
        # 아침 점심 저녁으로 나누기
        for t in ["m", "l", "d"]:
            # 데이터의 양이 극 소량이기 때문에 underfitting 방지를 위해 train과 test의 비율을
            # 99 : 1로 설정
            X_train, X_test, Y_train, Y_test = train_test_split(
                data[f"X{t}"], data[f"Y{t}"], test_size=0.01, random_state=42)
            # 스케일러 학습
            self.scaler.fit(X_train)
            # 학습 시작
            result.append(self.__machineLearning(self.scaler.transform(X_train), Y_train,
                                                 self.scaler.transform(X_test), Y_test, t))
        return result

    def __hyperParmSettingLR(self, model, X: pd.DataFrame, Y: pd.Series) -> None:
        parm_grid = {"alpha": list(np.arange(1.0, 11.0, 1.5)),
                     'max_iter': [1000, 1100, 1200, 1300],
                     'random_state': [132], 'tol': [1e-3]}
        gs = GridSearchCV(model, param_grid=parm_grid,
                          scoring='neg_mean_absolute_error',)
        # 최적의 파라미터 찾기
        gs.fit(X, Y)
        result = {"model": gs.best_estimator_, "parm": gs.best_params_}
        return result

    def __hyperParmSettingEN(self, model, X: pd.DataFrame, Y: pd.Series) -> None:
        parm_grid = {"alpha": list(np.arange(1.0, 11.0, 1.5)),
                     'max_iter': [1000, 1100, 1200, 1300],
                     "l1_ratio": [i for i in np.arange(0.0, 1.1, 0.1)]}
        gs = GridSearchCV(model, param_grid=parm_grid,
                          scoring='neg_mean_absolute_error',)
        # 최적의 파라미터 찾기
        gs.fit(X, Y)
        result = {"model": gs.best_estimator_, "parm": gs.best_params_}
        return result

    def __hyperParmSettingSVM(self, model, X: pd.DataFrame, Y: pd.Series) -> None:
        param_grid = {
            "kernel": ["linear", "poly", "rbf"],
            "C": [0.1, 1, 10],
            "epsilon": [0.1, 0.2, 0.3],
            'max_iter': [1000, 1100, 1200, 1300]
        }
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            # MAE를 적용하려면 'neg_mean_absolute_error'를 적용합니다.
            scoring='neg_mean_absolute_error',
            cv=5,
        )
        # 최적의 파라미터 찾기
        gs.fit(X, Y)
        result = {"model": gs.best_estimator_, "parm": gs.best_params_}
        return result

    def __hyperParmSettingXGBR(self, model, X: pd.DataFrame, Y: pd.Series) -> None:
        param_grid = {
            'n_estimators': [1000, 1100, 1200, 1300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'colsample_bytree': [0.6, 0.8, 1]
        }
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_absolute_error',
            cv=5,
        )
        # 최적의 파라미터 찾기
        gs.fit(X, Y)
        result = {"model": gs.best_estimator_, "parm": gs.best_params_}
        return result

    def __drawLearningCurve(self, model, title: str, Xtrain: pd.DataFrame, Ytrain: pd.Series, ylim=None, cv=10,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Accuracy")

        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=Xtrain,
            y=Ytrain,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes)
        # 평균, 표준 편차 계산
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)

        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # 시각화
        # Training score
        plt.plot(train_sizes, train_scores_mean, marker='o', markersize=5,
                 color='b', alpha=0.15, label='Training Score')
        # Cross-validation score
        plt.plot(train_sizes, test_scores_mean, linestyle='--', marker='s', markersize=5,
                 color='g', alpha=0.15, label='Cross-validation score')
        # 표준편차 연영 그리기
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean+train_scores_std, alpha=.2, color='r')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean+test_scores_std, alpha=.2, color='g')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"./Data/LearningCurve/{title.split('_')[0]}/{title}.png")

    def __machineLearning(self, Xtrain: pd.DataFrame, Ytrain: pd.Series, Xtest: pd.DataFrame, Ytest: pd.Series, t: str) -> dict:
        models = {"ridge": Ridge(), "lasso": Lasso(),
                  "EN": ElasticNet(), "SVM": SVR(), "XGBR": XGBRegressor()}
        result = {}
        for model in models.keys():
            if model == "ridge":
                result: dict = self.__hyperParmSettingLR(
                    models[model], Xtrain, Ytrain)
            elif model == "lasso":
                result: dict = self.__hyperParmSettingLR(
                    models[model], Xtrain, Ytrain)
            elif model == "EN":
                result: dict = self.__hyperParmSettingEN(
                    models[model], Xtrain, Ytrain)
            elif model == "SVM":
                result: dict = self.__hyperParmSettingSVM(
                    models[model], Xtrain, Ytrain)
            elif model == "XGBR":
                result: dict = self.__hyperParmSettingXGBR(
                    models[model], Xtrain, Ytrain)
            # 평가
            result["name"] = f"{model}_{t}"
            Ypred = result["model"].predict(Xtest)
            # mae
            mae = mean_absolute_error(
                Ytest, Ypred)
            result["MAE"] = int(mae)
            # 데이터 처리
            r = pd.concat([Ytest, pd.Series(
                map(int, Ypred), index=Ytest.index)], axis=1)
            r.columns = ["원본", "예측"]
            result["comparison"] = r
            # 학습 곡선
            self.__drawLearningCurve(
                result["model"], result["name"], Xtrain, Ytrain)
            with open(f"./Data/TrainingData/{result['name']}.pkl", "wb") as f:
                pickle.dump(result, f)
        return result

    def run(self) -> list:
        results = self.__splitTrainTestData(self.__featureSelector())
        return results


if __name__ == "__main__":
    mm = MachineLearningMate()
    result = mm.run()
