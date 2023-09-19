from processingAndVisual import *
from forecastMate import *
from predict import *


class PipeLine:
    def __init__(self, year, target) -> None:
        # 전처리 및 시각화 모듈
        self.pl = PreprocessingAndVisualization(year)
        # 모델 학습 모듈
        self.mm = MachineLearningMate()
        # 예측 모듈
        self.pr = Predict(target)

    def run(self):
        self.pl.run()
        self.mm.run()
        self.pr.run()
