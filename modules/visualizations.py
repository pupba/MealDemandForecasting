from itertools import groupby
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = './modules/HAMAUMGODIC.ttf'
font_prop = fm.FontProperties(fname=font_path, size=12)


class Visualization:
    def __init__(self, year) -> None:
        self.year = year

    # 1. 시간에 따른 식사 인원 변화

    def __drawTimeSeriesOfMealCount(self, df: pd.DataFrame) -> None:
        for i in df.loc[:, "년"].unique():
            # 년도별로 추출
            df_ = df[df["년"] == i]
            # 월별 총원 그룹화
            df_monthly = df_.groupby("월", as_index=False)["총원"].sum()
            # 그래프 크기
            plt.figure(figsize=(10, 5))
            # 그래프 그리기
            plt.bar(df_monthly.loc[:, "월"], df_monthly.loc[:, "총원"])
            # 제목
            plt.title(f"{i}년 월별 총 식사 인원", fontproperties=font_prop)
            # x 축 범위
            plt.xticks(df_monthly.loc[:, "월"], fontproperties=font_prop)
            # x,y 축 이름
            plt.xlabel("월", fontproperties=font_prop)
            plt.ylabel("총 식사 인원(명)", fontproperties=font_prop)
            # 바 그래프 위에 값 출력
            for x, y in zip(df_monthly.loc[:, "월"], df_monthly.loc[:, "총원"]):
                plt.text(x, y, str(y), ha="center", va="bottom")
            plt.savefig(
                f"./Data/Visualizations/시간에 따른 식사 인원 변화/{i}년 월별 총 식사 인원.png")

    # 2. 요일별 식사 인원

    def __drawWeekdayMealCount(self, df: pd.DataFrame) -> None:

        for i in df.loc[:, "년"].unique():
            # 년도별로 추출
            df_ = df[df["년"] == i]
            # 요일별 총원 그룹화
            df_weekday = df_.groupby("요일", as_index=False)["총원"].sum()
            df_weekday['요일'] = pd.Categorical(df_weekday['요일'], categories=[
                '월', '화', '수', '목', '금', '토', '일'], ordered=True)
            df_weekday = df_weekday.sort_values(by='요일')
            # 그래프 크기
            plt.figure(figsize=(10, 5))
            # 그래프 그리기
            plt.bar(df_weekday.loc[:, "요일"], df_weekday.loc[:, "총원"])
            # 제목
            plt.title(f"{i}년 월별 총 식사 인원", fontproperties=font_prop)
            # x 축 범위
            plt.xticks(df_weekday.loc[:, "요일"], fontproperties=font_prop)

            # x,y 축 이름
            plt.xlabel("요일", fontproperties=font_prop)
            plt.ylabel("총 식사 인원(명)", fontproperties=font_prop)
            # 바 그래프 위에 값 출력
            for x, y in zip(df_weekday.loc[:, "요일"], df_weekday.loc[:, "총원"]):
                plt.text(x, y, f"{str(y)}명", ha="center", va="bottom")
            plt.savefig(
                f"./Data/Visualizations/요일별 식사 인원/{i}년 요일별 총 식사 인원.png")

    # 3. 기온과 식사 인원의 관계

    def __drawTempToMealCountRelationship(self, df: pd.DataFrame) -> None:
        for i in df.loc[:, "년"].unique():
            # 년도별로 추출
            df_ = df[df["년"] == i]
            # 기온
            X = df_.loc[:, '평균기온(℃)']
            # 총원
            Y = df_.loc[:, "총원"]
            plt.figure(figsize=(10, 6))
            plt.xlabel('평균기온 (°C)', fontproperties=font_prop)
            plt.ylabel('총 식사 인원', fontproperties=font_prop)
            plt.title(f'{i}년 평균기온 별 총 식사 인원', fontproperties=font_prop)
            plt.scatter(X, Y, c='r', marker='o', label='식사 인원')
            plt.grid(True)
            legend = plt.legend()
            for text in legend.get_texts():
                text.set_fontproperties(font_prop)
            plt.savefig(
                f"./Data/Visualizations/기온과 식사 인원의 관계/{i}년 평균기온 별 총 식사 인원.png")

    # 4. 습도와 식사 인원의 관계

    def __drawHumidityToMealCountRelationship(self, df: pd.DataFrame) -> None:
        for i in df.loc[:, "년"].unique():
            # 년도별로 추출
            df_ = df[df["년"] == i]
            # 기온
            X = df_.loc[:, '평균습도(%rh)']
            # 총원
            Y = df_.loc[:, "총원"]
            plt.figure(figsize=(10, 6))
            plt.xlabel('평균습도(%rh)', fontproperties=font_prop)
            plt.ylabel('총 식사 인원', fontproperties=font_prop)
            plt.title(f'{i}년 평균습도 별 총 식사 인원', fontproperties=font_prop)
            plt.scatter(X, Y, c='r', marker='o', label='식사 인원')
            plt.grid(True)
            legend = plt.legend()
            for text in legend.get_texts():
                text.set_fontproperties(font_prop)
            plt.savefig(
                f"./Data/Visualizations/습도와 식사 인원의 관계/{i}년 평균습도 별 총 식사 인원.png")
    # 5. 불쾌한 상태와 식사 인원

    def __drawDiscomfortIndexAndMealCount(self, df: pd.DataFrame) -> None:
        for i in df.loc[:, "년"].unique():
            # 년도별로 추출
            df_ = df[df["년"] == i]
            # 불쾌지수가 체크된 행만 추출
            disFilter = df_['discomfort'] == True
            dfDiscom = df_[disFilter]
            nondfDiscom = df_[~disFilter]
            x1 = dfDiscom.loc[:, "총원"]
            y1 = dfDiscom.loc[:, "discomfort"]
            x2 = nondfDiscom.loc[:, "총원"]
            y2 = nondfDiscom.loc[:, "discomfort"]
            # 총원
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.hist(dfDiscom.loc[:, "총원"], bins=20, alpha=0.5)
            plt.xlabel('총 식사 인원', fontproperties=font_prop)
            plt.ylabel('불쾌지수 유무', fontproperties=font_prop)
            plt.title(f'{i}년 불쾌함을 느낄 때 식사인원', fontproperties=font_prop)

            plt.subplot(1, 2, 2)
            plt.hist(nondfDiscom.loc[:, "총원"], bins=20, alpha=0.5)
            plt.xlabel('총 식사 인원', fontproperties=font_prop)
            plt.ylabel('불쾌지수 유무', fontproperties=font_prop)
            plt.title(f'{i}년 불쾌함을 느끼지 않을 때 식사인원', fontproperties=font_prop)
            plt.grid(True)
            plt.savefig(
                f"./Data/Visualizations/불쾌한 상태와 식사 인원/{i}년 불쾌한 상태 유무에 대한 식사인원 분포.png")

    # 6. 행사 여부와 식사 인원

    def __categorizeEvent(self, event: str) -> str:
        if event == "없음":
            return "없음"
        elif "휴일" in event:
            return "휴일"
        else:
            return "행사"

    def __drawMealCountAndEvent(self, df: pd.DataFrame) -> None:
        for i in df.loc[:, "년"].unique():
            # 년도별로 추출
            df_ = df[df["년"] == i]
            df_.loc[:, "행사"] = df_.loc[:, "행사"].apply(self.__categorizeEvent)
            # 그리기
            plt.figure(figsize=(10, 5))
            eType = df_.loc[:, "행사"].unique()
            for j, Type in enumerate(eType):
                plt.subplot(1, len(eType), j+1)
                plt.boxplot(df_[df_.loc[:, '행사'] == Type]["총원"])
                plt.title(f'{i}년 이벤트 종류에 따른 총 식사 인원', fontproperties=font_prop)
                plt.xlabel(f'{Type}', fontproperties=font_prop)
                plt.ylabel('총 식사 인원', fontproperties=font_prop)
            plt.tight_layout()
            plt.savefig(
                f"./Data/Visualizations/행사 여부와 식사 인원/{i}년 이벤트 종류에 따른 총 식사 인원.png")

    # 7. 강수량과 식사 인원

    def __categorizeRainfall(self, rainfall: float) -> str:
        if rainfall == 0:
            return '비 없음'
        elif rainfall < 3*24:
            return '약한 비'
        elif rainfall < 15*24:
            return '보통 비'
        elif rainfall < 30*24:
            return '강한 비'
        else:
            return '매우 강한 비'

    def __drawMealCountToPrecipitation(self, df: pd.DataFrame) -> None:
        for i in df.loc[:, "년"].unique():
            # 년도별로 추출
            df_ = df[df["년"] == i]
            df__ = df_.loc[:, ["강수량(mm)", "총원"]]
            """
            약한 비는 1시간에 3mm 미만
            보통 비는 1시간에 3~15mm 미만 
            강한 비는 1시간에 15mm 이상 
            매우 강한 비는 1시간에 30mm 이상
            일강수량 데이터이기 때문에 그룹을 시간당 강수량이 일정하게 온다고 가정하고 나눔
            """
            df__.loc[:, "구분"] = df__.loc[:, "강수량(mm)"].apply(
                self.__categorizeRainfall)
            rainGroup = df__.groupby("구분", as_index=False)["총원"].sum()
            # 그리기
            plt.figure(figsize=(10, 5))
            # 그래프 그리기
            plt.bar(rainGroup.loc[:, "구분"], rainGroup.loc[:, "총원"])
            # 제목
            plt.title(f"{i}년 강우 종류별 총 식사 인원", fontproperties=font_prop)
            # x축
            plt.xticks(rainGroup.loc[:, "구분"].unique(),
                       fontproperties=font_prop)
            # x,y 축 이름
            plt.xlabel("강우 종류", fontproperties=font_prop)
            plt.ylabel("총 식사 인원(명)", fontproperties=font_prop)
            # 바 그래프 위에 값 출력
            for x, y in zip(rainGroup.loc[:, "구분"], rainGroup.loc[:, "총원"]):
                plt.text(x, y, f"{str(y)}명", ha="center",
                         va="bottom", fontproperties=font_prop, fontsize=10)
            plt.savefig(
                f"./Data/Visualizations/강우량과 식사 인원/{i}년 강우 종류별 식사 인원.png")

    # 8. 메뉴와 식사 인원 간의 관계

    def __drawMenuKindMealCount(self, df: pd.DataFrame) -> None:
        def flatten(l: list):
            result = []
            for item in l:
                result.extend(item)
            return result
        for i in df.loc[:, "년"].unique():
            # 년도별로 추출
            df_ = df[df["년"] == i]
            for j in ["조식", "중식", "석식"]:
                df__ = [df_.loc[:, f"{j}반찬{k}"].to_list() for k in range(1, 4)]
                foods = flatten(df__)
                # 결측치 처리
                foods = pd.DataFrame(sorted([x for x in foods if not (
                    isinstance(x, float) and np.isnan(x))]), columns=["음식"])
                # 그룹화
                foodGroup = foods.groupby("음식").size()
                top5 = foodGroup.nlargest(5)
                # 시각화
                plt.figure(figsize=(14, 7))
                top5.plot(kind="bar")
                plt.title(f"{i}년 {j} Top5 메뉴 빈도수", fontproperties=font_prop)
                plt.xlabel("음식명", fontproperties=font_prop)
                plt.xticks(rotation=0, fontproperties=font_prop)
                plt.ylabel("빈도수", fontproperties=font_prop)
                for h, v in enumerate(top5):
                    plt.text(h-0.1, v+1, f"{v}번", fontproperties=font_prop)
                plt.savefig(
                    f"./Data/Visualizations/메뉴와 식사인원의 관계/{i}년 {j} 메뉴 Top5 빈도수.png")
            # 가장 식사 인원이 많은 날에 메뉴
            top5M = df_.nlargest(5, '조식 인원')[
                ['조식국', '조식반찬1', '조식반찬2', '조식반찬3', '조식 인원']]
            tmp = top5M.loc[:, ['조식국', '조식반찬1', '조식반찬2', '조식반찬3']].apply(
                lambda row: ','.join(list(set(row.ffill(axis=0)))), axis=1)
            top5M = pd.DataFrame({"메뉴": tmp, "인원": top5M.loc[:, "조식 인원"]})
            top5L = df_.nlargest(5, '중식 인원')[
                ['중식국', '중식반찬1', '중식반찬2', '중식반찬3', '중식 인원']]
            tmp = top5L.loc[:, ['중식국', '중식반찬1', '중식반찬2', '중식반찬3']].apply(
                lambda row: ','.join(list(set(row.ffill(axis=0)))), axis=1)
            top5L = pd.DataFrame({"메뉴": tmp, "인원": top5L.loc[:, "중식 인원"]})
            top5D = df_.nlargest(5, '석식 인원')[
                ['석식국', '석식반찬1', '석식반찬2', '석식반찬3', '석식 인원']]
            tmp = top5D.loc[:, ['석식국', '석식반찬1', '석식반찬2', '석식반찬3']].apply(
                lambda row: ','.join(list(set(row.ffill(axis=0)))), axis=1)
            top5D = pd.DataFrame({"메뉴": tmp, "인원": top5D.loc[:, "석식 인원"]})

            fig, axes = plt.subplots(3, 1, figsize=(30, 15))

            # 조식 데이터 시각화
            axes[0].barh(top5M['메뉴'], top5M['인원'], color='green')
            axes[0].set_title('Top 5 조식 식사 인원과 메뉴', fontproperties=font_prop)
            axes[0].set_xlabel('식사 인원', fontproperties=font_prop)
            axes[0].set_yticklabels(top5M['메뉴'], fontproperties=font_prop)

            for y, v in enumerate(top5M['인원']):
                axes[0].text(v + 3, y, f'{v}명', color='black',
                             va='center', fontproperties=font_prop)

            # 중식 데이터 시각화
            axes[1].barh(top5L['메뉴'], top5L['인원'], color='blue')
            axes[1].set_title('Top 5 중식 식사 인원과 메뉴', fontproperties=font_prop)
            axes[1].set_xlabel('식사 인원', fontproperties=font_prop)
            axes[1].set_yticklabels(top5L['메뉴'], fontproperties=font_prop)

            for y, v in enumerate(top5L['인원']):
                axes[1].text(v + 3, y, f'{v}명', color='black',
                             va='center', fontproperties=font_prop)

            # 석식 데이터 시각화
            axes[2].barh(top5D['메뉴'], top5D['인원'], color='orange')
            axes[2].set_title('Top 5 석식 식사 인원과 메뉴', fontproperties=font_prop)
            axes[2].set_xlabel('식사 인원', fontproperties=font_prop)
            axes[2].set_yticklabels(top5D['메뉴'], fontproperties=font_prop)

            for y, v in enumerate(top5D['인원']):
                axes[2].text(v + 3, y, f'{v}명', color='black',
                             va='center', fontproperties=font_prop)

            # 서브플롯 간 간격 조정
            plt.tight_layout()

            # 그래프 출력
            plt.savefig(
                f"./Data/Visualizations/메뉴와 식사인원의 관계/{i}년 TOP5 메뉴와 식사 인원.png")

    # data load

    def __loadData(self) -> pd.DataFrame:
        PATH: str = "./Data/ProcessedData/"
        meal: pd.DataFrame = pd.read_excel(PATH+f"Processed_{self.year}MealDATA.xlsx",
                                           engine="openpyxl", na_values=np.NaN)
        meal['일시'] = pd.to_datetime(meal['년'].astype(
            str) + '-' + meal['월'].astype(str) + '-' + meal['일'].astype(str))
        weather: pd.DataFrame = pd.read_excel(
            PATH+f"Processed_{self.year}WeatherDATA.xlsx", engine="openpyxl", na_values=np.NaN)
        # 데이터 병합
        total: pd.DataFrame = pd.merge(meal, weather, on='일시', how="inner")
        return total

    def run(self) -> None:
        df: pd.DataFrame = self.__loadData()
        # 1. 시간에 따른 식사 인원 변화
        self.__drawTimeSeriesOfMealCount(df)
        # 2. 요일별 식사 인원
        self.__drawWeekdayMealCount(df)
        # 3. 기온과 식사 인원의 관계
        self.__drawTempToMealCountRelationship(df)
        # 4. 습도와 식사 인원의 관계
        self.__drawHumidityToMealCountRelationship(df)
        # 5. 불쾌한 상태와 식사 인원
        self.__drawDiscomfortIndexAndMealCount(df)
        # 6. 행사 여부와 식사 인원
        self.__drawMealCountAndEvent(df)
        # 7. 강수량과 식사 인원
        self.__drawMealCountToPrecipitation(df)
        # 8. 메뉴와 식사 인원 간의 관계
        self.__drawMenuKindMealCount(df)


if __name__ == "__main__":
    pass
