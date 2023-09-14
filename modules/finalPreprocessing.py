import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

font_path = './modules/HAMAUMGODIC.ttf'
font_prop = fm.FontProperties(fname=font_path, size=12)


class FinalPreprocessing:
    def __init__(self, year) -> None:
        self.year = year
    # 식당 데이터 + 날씨 데이터

    def __mergeData(self, year: int) -> pd.DataFrame:
        try:
            PATH: str = "./Data/ProcessedData/"
            meal: pd.DataFrame = pd.read_excel(PATH+f"Processed_{year}MealDATA.xlsx",
                                               engine="openpyxl", na_values=np.NaN)
            meal['일시'] = pd.to_datetime(meal['년'].astype(
                str) + '-' + meal['월'].astype(str) + '-' + meal['일'].astype(str))
            weather: pd.DataFrame = pd.read_excel(
                PATH+f"Processed_{year}WeatherDATA.xlsx", engine="openpyxl", na_values=np.NaN)
            # 데이터 병합
            total: pd.DataFrame = pd.merge(meal, weather, on='일시', how="inner")
            # 결측치 처리
            total = total.fillna("no")
            # 최고 기온 시각, 최저 기온 시각, 일시, 식사 타입 삭제
            total = total.iloc[:, list(range(0, 21)) +
                               list(range(22, 27))+[28, 30]]
            return total
        except FileNotFoundError:
            print("파일이 없습니다.")

    def __dataEncoding(self, df: pd.DataFrame) -> pd.DataFrame:
        # 컬럼 이름 영어로 변경
        reOrder = {'년': "Year", '월': "Month", '일': "Day", '요일': "Day of Week",
                   '조식 인원': "Breakfast Attendance", '조식국': "Breakfast Soup",
                   '조식반찬1': "Breakfast Dish1", '조식반찬2': "Breakfast Dish2", '조식반찬3': "Breakfast Dish3",
                   '중식 인원': "Lunch Attendance", '중식국': "Lunch Soup",
                   '중식반찬1': "Lunch Dish1", '중식반찬2': "Lunch Dish2", '중식반찬3': "Lunch Dish3",
                   '석식 인원': "Dinner Attendance", '석식국': "Dinner Soup",
                   '석식반찬1': "Dinner Dish1", '석식반찬2': "Dinner Dish2", '석식반찬3': "Dinner Dish3",
                   '총원': "Total Attendance", '행사': "Event", '강수량(mm)': "Precipitation",
                   '평균습도(%rh)': "Average Humidity", '최저습도(%rh)': "Minimum Humidity",
                   '평균기온(℃)': "Average Temperature", '최고기온(℃)': "Maximum Temperature",
                   '최저기온(℃)': "Minimum Temperature", 'discomfort': "Discomfort Index", 'mealType': "Meal Type"}
        df.rename(columns=reOrder, inplace=True)
        # 라벨 인코딩
        lb = LabelEncoder()
        df_ = df.copy()
        # object인 컬럼만 인코딩
        object_columns = df_.select_dtypes(include=['object']).columns
        for col in object_columns:
            df_[col] = lb.fit_transform(df_[col])

        # 원핫 인코딩
        df_encoded = pd.get_dummies(df, columns=object_columns)
        return df_, df_encoded

    def __drawCorrelationCoefficient(self, df: pd.DataFrame, year: int) -> None:
        for time, rg in [("조식", (4, 9)), ("중식", (9, 14)), ("석식", (14, 19))]:
            # year는 다른 컬럼과 상관관계를 찾기 어려움으로 제외
            data = df.iloc[:, [1, 2, 3] +
                           list(range(rg[0], rg[1]))+list(range(19, 28))]
            correlation_matrix = data.corr()
            plt.figure(figsize=(15, 10))
            sns.heatmap(correlation_matrix, annot=True)
            plt.title(f"{year}년, {time} 데이터 컬럼간의 상관관계 매트릭스",
                      fontproperties=font_prop)
            plt.tight_layout()
            plt.savefig(
                f"./Data/Visualizations/상관관계분석/{year}년, {time} 데이터 컬럼간의 상관관계 분석.png")

    def run(self) -> None:
        df: pd.DataFrame = self.__mergeData(self.year)
        # 데이터 인코딩
        labelEC, oneHotEC = self.__dataEncoding(df)
        self.__drawCorrelationCoefficient(labelEC, self.year)
        # 데이터 저장
        SAVEPATHLB: str = f"./Data/CleanedData/CleanedData_{year}_label.xlsx"
        labelEC.to_excel(SAVEPATHLB, sheet_name="학습을 위한 데이터", index=False)
        # 데이터 저장
        SAVEPATHOH: str = f"./Data/CleanedData/CleanedData_{year}_onehot.xlsx"
        oneHotEC.to_excel(SAVEPATHOH, sheet_name="학습을 위한 데이터", index=False)


if __name__ == "__main__":
    for year in [2018, 2019]:
        fp = FinalPreprocessing(year)
        fp.run()
