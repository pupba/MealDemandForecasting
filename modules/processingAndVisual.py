from mealDataProcessed import *
from weatherDataProcessed import *
from visualizations import *
from finalPreprocessing import *
from forecastMate import *


class PreprocessingAndVisualization:
    def __init__(self, year: str) -> None:
        self.__year = year
        # 식당 데이터 처리 모듈
        self.pdmm = ProcessedMDataModule(self.__year)
        # 날씨 데이터 처리 모듈
        self.pdw = ProcessedWDataModule(self.__year)
        # 시각화 모듈
        self.visual = Visualization(self.__year)
        # 최종 데이터 처리 모듈
        self.final = FinalPreprocessing(year)

    def run(self) -> None:
        # 식당 데이터 처리
        self.pdmm.run()
        # 날씨 데이터 처리
        self.pdw.run()
        # 시각화
        self.visual.run()
        # 최종 데이터 처리
        self.final.run()


if __name__ == "__main__":
    pl = PreprocessingAndVisualization(2023)
    pl.run()
