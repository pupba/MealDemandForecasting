from modules.mealDataProcessed import *
from modules.weatherDataProcessed import *
from modules.visualizations import *
if __name__ == "__main__":
    # pdwm = ProcessedWDataModule("2018~2019")
    # pdwm.run()
    for f in ["2018", "2019"]:
        pdmm = ProcessedMDataModule(f)
        pdmm.run()
        vs = Visualization(f)
        vs.run()
    pass
