import pandas as pd
import numpy as np
from typing import Union
import re
""" 1차 가공 데이터 저장을 위한 모듈"""


class ProcessedMDataModule:
    def __init__(self, fileName: str) -> None:
        self.__fileName = fileName

    # 결측치 처리
    def __processNanValues(self, df: pd.DataFrame) -> Union[pd.DataFrame, list]:
        # 결측치가 있는 컬럼명 확인
        missingValueCol: list = df.columns[df.isnull().any()].tolist()
        # 결측치가 존재하는 행 삭제
        df_: pd.DataFrame = df.dropna()
        return df_, missingValueCol

    # 날짜 데이터 처리
    def __splitDate(self, df: pd.DataFrame) -> pd.DataFrame:
        # 날짜열을 추출
        date: pd.Series = df.iloc[:, 0]
        # 년,월,일로 분할
        year: pd.Series = date.dt.year
        month: pd.Series = date.dt.month
        day: pd.Series = date.dt.day
        # 기존 데이터 프레임에 병합
        df_ = df.drop(["날짜"], axis=1)
        df_.insert(0, "년", year)
        df_.insert(1, "월", month)
        df_.insert(2, "일", day)
        return df_

    # 메뉴 처리

    def __menuProcessing(self, df: pd.DataFrame) -> pd.DataFrame:
        # 메뉴 컬럼 추출
        # 아침
        morning: pd.Series = df.iloc[:, 4].apply(
            lambda x: re.split(r'[!@#^&*.,/]', x))
        # 점심
        lunch: pd.Series = df.iloc[:, 6].apply(
            lambda x: re.split(r'[!@#^&*.,/]', x))
        # 저녁
        dinner: pd.Series = df.iloc[:, 8].apply(
            lambda x: re.split(r'[!@#^&*.,/]', x))

        """ 영향을 미치는 메뉴는 일반적으로 1,2,3에 해당하는 음식들과 마지막에 부식의 유무에 따라 결정된다고 가정"""
        # 메뉴중 첫번째가 일반적으로 국
        soupM: pd.Series = morning.apply(
            lambda x: x[0] if len(x) > 0 else None)
        soupL: pd.Series = lunch.apply(lambda x: x[0] if len(x) > 0 else None)
        soupD: pd.Series = dinner.apply(lambda x: x[0] if len(x) > 0 else None)
        # 메뉴중 두번째,세번쨰가 메인반찬
        menu1M: pd.Series = morning.apply(
            lambda x: x[1] if len(x) > 1 else None)
        menu1L: pd.Series = lunch.apply(lambda x: x[1] if len(x) > 1 else None)
        menu1D: pd.Series = dinner.apply(
            lambda x: x[1] if len(x) > 1 else None)
        menu2M: pd.Series = morning.apply(
            lambda x: x[2] if len(x) > 2 else None)
        menu2L: pd.Series = lunch.apply(lambda x: x[2] if len(x) > 2 else None)
        menu2D: pd.Series = dinner.apply(
            lambda x: x[2] if len(x) > 2 else None)
        # 일반적으로 메뉴중 마지막이 부식, 부식이 아닌경우는 그냥 음식으로 대체
        menu3M: pd.Series = morning.apply(
            lambda x: x[-1] if len(x) > -1 else None)
        menu3L: pd.Series = lunch.apply(
            lambda x: x[-1] if len(x) > -1 else None)
        menu3D: pd.Series = dinner.apply(
            lambda x: x[-1] if len(x) > -1 else None)

        df_: pd.DataFrame = pd.DataFrame({
            "조식국": soupM,
            "조식반찬1": menu1M,
            "조식반찬2": menu2M,
            "조식반찬3": menu3M,
            "중식국": soupL,
            "중식반찬1": menu1L,
            "중식반찬2": menu2L,
            "중식반찬3": menu3L,
            "석식국": soupD,
            "석식반찬1": menu1D,
            "석식반찬2": menu2D,
            "석식반찬3": menu3D,
        })
        df___: pd.DataFrame = df.drop(columns=["조식 메뉴", "중식 메뉴", "석식 메뉴"])
        df__: pd.DataFrame = pd.concat([df___, df_], axis=1)
        return df__

    def run(self) -> None:
        PATH: str = "./OriginData/meal/"
        try:
            df18 = pd.read_excel(PATH+self.__fileName,
                                 engine="openpyxl", na_values=np.NaN)
        except FileNotFoundError:
            print("파일이 없습니다.")
        # 1. 결측치 처리
        dfNaOK, missValCol = self.__processNanValues(df18)
        if len(missValCol) != 0:
            print(f"결측치가 존재한 열 ->{missValCol}")
        else:
            # 2. 날짜 처리
            dfDateOK: pd.DataFrame = self.__splitDate(dfNaOK)
            # 3. 메뉴 처리
            dfMenuOK: pd.DataFrame = self.__menuProcessing(dfDateOK)
            # 4. reindex
            newOrder: list = ['년', '월', '일', '요일',
                              '조식 인원', '조식국', '조식반찬1', '조식반찬2', '조식반찬3',
                              '중식 인원', '중식국', '중식반찬1', '중식반찬2', '중식반찬3',
                              '석식 인원', '석식국', '석식반찬1', '석식반찬2', '석식반찬3',
                              '총원', '비고',]
            df_: pd.DataFrame = dfMenuOK[newOrder]
            SAVEPATH: str = f"./Data/ProcessedData/Processed_{self.__fileName}"
            df_.to_excel(SAVEPATH, sheet_name="1차 가공 식당데이터", index=False)


if __name__ == "__main__":
    pass
