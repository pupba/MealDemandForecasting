import pandas as pd
import numpy as np


class ProcessedWDataModule:
    def __init__(self, fileName: str) -> None:
        self.__fileName = fileName

    def __processNanValue(self, df: pd.DataFrame) -> pd.DataFrame:
        # 파일에서 강수량 부분만 NaN가 있음.
        # 강수량에서 NaN은 비가 안온것이기 때문에 0으로 대체
        df: pd.DataFrame = df.fillna(float(0))
        return df

    def __mergeData(self, *DataFrames: pd.DataFrame) -> pd.DataFrame:
        # 병합
        dfWeather: pd.DataFrame = pd.DataFrame()
        for df in DataFrames:
            if dfWeather.empty:
                dfWeather = df
            else:
                dfWeather = pd.merge(dfWeather, df, on="일시", how="inner")
        return dfWeather

    def __discomfort(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            식사 시간
            조식 : 07:00 ~ 08:00
            중식 : 11:30 ~ 13:00
            석식 : 17:30 ~ 18:30 
        """
        # 식사 시간대 설정
        morning_start = pd.to_datetime("07:00:00", format="%H:%M:%S").time()
        morning_end = pd.to_datetime("08:00:00", format="%H:%M:%S").time()
        lunch_start = pd.to_datetime("11:30:00", format="%H:%M:%S").time()
        lunch_end = pd.to_datetime("13:00:00", format="%H:%M:%S").time()
        dinner_start = pd.to_datetime("17:30:00", format="%H:%M:%S").time()
        dinner_end = pd.to_datetime("18:30:00", format="%H:%M:%S").time()
        # 여름 5월 ~ 10월
        df['month'] = df["일시"].dt.month
        summer = df[(df['month'] >= 5) & (df['month'] <= 10)]
        # 문자열을 시간으로
        summer.loc[:, '최고기온시각'] = pd.to_datetime(
            summer.loc[:, '최고기온시각'], format="%H:%M:%S").dt.time
        # 시간대에 속하는 데이터와 그렇지 않은 데이터로 분리
        mealTime_morning = summer[((summer.loc[:, '최고기온시각'] >= morning_start) & (
            summer.loc[:, '최고기온시각'] <= morning_end))]
        mealTime_lunch = summer[((summer.loc[:, '최고기온시각'] >= lunch_start) & (
            summer.loc[:, '최고기온시각'] <= lunch_end))]
        mealTime_dinner = summer[((summer.loc[:, '최고기온시각'] >= dinner_start) & (
            summer.loc[:, '최고기온시각'] <= dinner_end))]
        noTime = summer[~summer.index.isin(
            mealTime_lunch.index)]
        noTime = noTime[~noTime.index.isin(
            mealTime_dinner.index)]
        """
            🐶 불쾌지수 체크 : 불쾌지수는 기온과 습도를 조합해 나타낼 수 있다.
            여름 기간 체크
            🐱 68 이상이면 불쾌지수 체크
            최고기온 : 시각이 식사 시간(아침,점심,저녁) 시간일 경우 체크
            최저기온 : 시각이 식사 시간(아침,점심,저녁) 시간일 경우 체크
            평균습도 : 평균 습도가 불쾌지수 기준보다 높을 경우 체크
            🐨 Thom 불쾌지수=1.8x기온–0.55x(1–습도)x(1.8x기온–26)+32
        """

        # 불쾌지수 체크
        mealTime_morning['discomfort'] = mealTime_morning.apply(
            self.__calDiscomfort, axis=1)
        mealTime_morning['mealType'] = "breakfast"
        mealTime_lunch['discomfort'] = mealTime_lunch.apply(
            self.__calDiscomfort, axis=1)
        mealTime_lunch['mealType'] = "lunch"
        mealTime_dinner['discomfort'] = mealTime_dinner.apply(
            self.__calDiscomfort, axis=1)
        mealTime_dinner['mealType'] = "dinner"
        noTime['discomfort'] = False
        noTime['mealType'] = "None"
        # summer 데이터 병합
        summer = pd.concat([mealTime_lunch, mealTime_dinner, noTime], axis=0)
        # 겨울 11월 ~ 3월
        winter = df[(df['month'] >= 11) | (df['month'] <= 4)]
        # 문자열을 시간으로
        winter.loc[:, '최저기온시각'] = pd.to_datetime(
            winter.loc[:, '최저기온시각'], format="%H:%M:%S").dt.time
        # 시간대에 속하는 데이터와 그렇지 않은 데이터로 분리
        mealTime_morning = winter[((winter.loc[:, '최저기온시각'] >= morning_start) & (
            winter.loc[:, '최저기온시각'] <= morning_end))]
        mealTime_lunch = winter[((winter.loc[:, '최저기온시각'] >= lunch_start) & (
            winter.loc[:, '최저기온시각'] <= lunch_end))]
        mealTime_dinner = winter[((winter.loc[:, '최저기온시각'] >= dinner_start) & (
            winter.loc[:, '최저기온시각'] <= dinner_end))]
        noTime = winter[~winter.index.isin(mealTime_lunch.index)]
        noTime = noTime[~noTime.index.isin(mealTime_dinner.index)]
        """
            겨울철에는 12도 이하가 되면 춥다고 느낌
        """
        # 최저 기온이 12도 이하면 체크
        mealTime_morning['discomfort'] = mealTime_morning.apply(
            self.__checkTemperature, axis=1)
        mealTime_morning['mealType'] = "breakfast"
        mealTime_lunch['discomfort'] = mealTime_lunch.apply(
            self.__checkTemperature, axis=1)
        mealTime_lunch['mealType'] = "lunch"
        mealTime_dinner['discomfort'] = mealTime_dinner.apply(
            self.__checkTemperature, axis=1)
        mealTime_dinner['mealType'] = "dinner"
        noTime['discomfort'] = False
        noTime['mealType'] = "None"
        # winter 데이터 병합
        winter = pd.concat([mealTime_lunch, mealTime_dinner, noTime], axis=0)

        # summer, winter 병합
        weather = pd.concat([summer, winter], axis=0)
        # 월 열 삭제
        weather = weather.drop(['month'], axis=1)
        # winter 부분에 숫자로 표시되는 경우가 있음 -> 명시적 Boolean으로 변경
        weather['discomfort'] = weather['discomfort'].astype(bool)
        return weather

    def __calDiscomfort(self, row: pd.Series) -> bool:
        # 계산 공식 출처
        # https://www.nhimc.or.kr/ilsan_news/Hello_2018Summer/html/sub01_03.html
        temp = row.loc["최고기온(℃)"]
        humi = row.loc["평균습도(%rh)"]
        discomfortRate = 1.8 * temp - 0.55 * (1-humi) * (1.8*temp-26)+32
        if discomfortRate > 65:  # 불쾌감 느낌
            return True
        else:  # 불쾌감 느끼지 않음
            return False

    def __checkTemperature(self, row: pd.Series) -> bool:
        temp = row.loc["최저기온(℃)"]
        if temp <= 12.0:
            return True
        else:
            return False

    def run(self) -> None:
        PATH: str = "./OriginData/weather/"
        try:
            dfRainfall: pd.DataFrame = pd.read_excel(
                PATH+f"{self.__fileName} 강수량.xlsx", engine="openpyxl", na_values=np.NaN)
            dfTemperature: pd.DataFrame = pd.read_excel(
                PATH+f"{self.__fileName} 기온.xlsx", engine="openpyxl", na_values=np.NaN)
            dfHumidity: pd.DataFrame = pd.read_excel(
                PATH+f"{self.__fileName} 습도.xlsx", engine="openpyxl", na_values=np.NaN)
        except FileNotFoundError:
            print("파일이 없습니다...")
        # 필요 컬럼만 추출
        # 1. 강수량
        dfRainfall_: pd.DataFrame = dfRainfall.loc[:, ["일시", "강수량(mm)"]]
        # 2. 기온
        dfTemperature_: pd.DataFrame = dfTemperature.loc[:, [
            "일시", "평균기온(℃)", "최고기온(℃)", "최고기온시각", "최저기온(℃)", "최저기온시각"]]
        # 3. 습도
        dfHumidity_: pd.DataFrame = dfHumidity.loc[:, [
            "일시", "평균습도(%rh)", "최저습도(%rh)"]]
        # 병합
        dfWeather: pd.DataFrame = self.__mergeData(
            dfRainfall_, dfHumidity_, dfTemperature_)

        # 결측치 처리
        dfWeather: pd.DataFrame = self.__processNanValue(dfWeather)
        # 불쾌지수
        dfDiscomfort: pd.DataFrame = self.__discomfort(dfWeather)
        # 정렬
        dfDiscomfort = dfDiscomfort.sort_index()

        # 엑셀 파일로 만들기
        SAVEPATH: str = f"./Data/ProcessedData/Processed_{self.__fileName}WeatherData.xlsx"
        dfDiscomfort.to_excel(SAVEPATH, sheet_name="1차 가공 날씨 데이터", index=False)


if __name__ == "__main__":
    pass
