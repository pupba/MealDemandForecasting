from sklearn.preprocessing import StandardScaler
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


class Predict:
    def __init__(self, filename) -> None:
        self.__filename = filename

    def __loadModel(self) -> dict:
        models = {"breakfast": [], "lunch": [], "dinner": []}
        for data in os.listdir("./Data/TrainingData"):
            if data != ".DS_Store":
                time: str = data.split("_")[1].split(".")[0]
                with open(f"./Data/TrainingData/{data}", "rb") as f:
                    dic = pickle.load(f)
                if time == "m":  # 아침 모델
                    models["breakfast"].append((dic["name"], dic["model"]))
                elif time == "l":  # 점심 모델
                    models["lunch"].append((dic["name"], dic["model"]))
                elif time == "d":  # 자녁 모델
                    models["dinner"].append((dic["name"], dic["model"]))
        return models

    def __loadData(self) -> None:
        PATH: str = f"./Data/ForPredict/{self.__filename}"
        data = pd.read_excel(PATH, engine="openpyxl")

        # 아침 점심 저녁으로 나누고 X,Y로 나누기
        label = {"Xm": None, "Ym": None, "Xl": None,
                 "Yl": None, "Xd": None, "Yd": None}
        # 독립 변수 - Feature
        for rg in [("m", 5, 9), ("l", 10, 14), ("d", 14, 19)]:
            label[f"X{rg[0]}"] = data.iloc[:, [1, 2, 3] +
                                           list(range(rg[1], rg[2]))+list(range(19, 28))]
        # 종속 변수 - Target
        for tg in [("m", "Breakfast"), ("l", "Lunch"), ("d", "Dinner")]:
            label[f"Y{tg[0]}"] = data.loc[:, f"{tg[1]} Attendance"]

        # 스케일러
        sc = StandardScaler()
        maes = []
        # 엑셀 저장을 위한 엑셀 파일 생성
        writer = pd.ExcelWriter(
            f"./Data/PredictData/{self.__filename.split('_')[1]}.xlsx")
        # 데이터 처리
        for t in ["m", "l", "d"]:
            # 스케일러 학습
            X_scaled = sc.fit_transform(label[f"X{t}"])
            # 예측
            models = self.__loadModel()
            if t == "m":
                for model in models["breakfast"]:
                    # 예측
                    pred = model[1].predict(X_scaled)
                    # mae
                    mae = mean_absolute_error(
                        label[f"Y{t}"], pred)
                    maes.append(f"{model[0]}_MAE : {int(mae)}명")
                    # 결과 데이터 프레임화
                    r = pd.concat([label[f"Y{t}"], pd.Series(
                        map(int, pred), index=label[f"Y{t}"].index)], axis=1)
                    r.columns = ["원본", "예측"]
                    r["오차"] = np.abs(r.loc[:, "원본"]-r.loc[:, "예측"])
                    r.to_excel(
                        writer, sheet_name=f"{model[0]}_Result", index=False)
            elif t == "l":
                for model in models["lunch"]:
                    # 예측
                    pred = model[1].predict(X_scaled)
                    # mae
                    mae = mean_absolute_error(
                        label[f"Y{t}"], pred)
                    maes.append(f"{model[0]}_MAE : {int(mae)}명")
                    # 결과 데이터 프레임화
                    r = pd.concat([label[f"Y{t}"], pd.Series(
                        map(int, pred), index=label[f"Y{t}"].index)], axis=1)
                    r.columns = ["원본", "예측"]
                    r["오차"] = np.abs(r.loc[:, "원본"]-r.loc[:, "예측"])
                    r.to_excel(
                        writer, sheet_name=f"{model[0]}_Result", index=False)
            elif t == "d":
                for model in models["dinner"]:
                    # 예측
                    pred = model[1].predict(X_scaled)
                    # mae
                    mae = mean_absolute_error(
                        label[f"Y{t}"], pred)
                    maes.append(f"{model[0]}_MAE : {int(mae)}명")
                    # 결과 데이터 프레임화
                    r = pd.concat([label[f"Y{t}"], pd.Series(
                        map(int, pred), index=label[f"Y{t}"].index)], axis=1)
                    r.columns = ["원본", "예측"]
                    r["오차"] = np.abs(r.loc[:, "원본"]-r.loc[:, "예측"])
                    r.to_excel(
                        writer, sheet_name=f"{model[0]}_Result", index=False)
        writer.close()
        # MAE 값 저장
        with open(f"./Data/PredictData/Result_MAE.txt", "wt") as f:
            f.write("\n".join(maes))

    def run(self):
        self.__loadData()


if __name__ == "__main__":
    pr = Predict("CleanedData_2023_label.xlsx")
    pr.run()
