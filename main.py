import cv2
import pathlib, pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from cleaning_preprocess import ImageProcessor

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename



HOME = pathlib.Path().resolve().parent /"Malaria"
IMG_DIR = HOME / 'Images'
DATASETS = HOME / 'Datasets'
PARASITISED = IMG_DIR / 'Parasitized'
UNINFECTED = IMG_DIR / 'Uninfected'
MODELS = HOME / 'Models'


def latest_file(path=MODELS):
    files = path.glob("*.pickle")
    return max(files, key=lambda x: x.stat().st_ctime)

class MalariaModel():
    def __init__(self, train=False, random_state=None):
        self.df = pd.read_csv(DATASETS / "malaria_dataset.csv")
        self.df = self.df.sort_values("Label")
        self.y_cols = "Label"
        self.x_cols = [x for x in self.df.columns if x!=self.y_cols]

        self.X = self.df[self.x_cols]
        self.y = self.df[self.y_cols]


        if train:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=random_state)
            self.train_model()
        else:
            print("Since train == False, latest model will be loaded!")
            model_to_load = input("Enter Model name to load, or press Enter for latest model: ")
            if model_to_load=="":
                self.load_model()
                return  print("Loaded the latest model")
            try:
                self.trained_model = self.load_model(model_name=MODELS/model_to_load)
                return print("Model loaded Successfully")
            except Exception as e:
                print(e)
                return print("Failed to load model! Check model name... Models can be found in Models folder.")
            
    def train_model(self):
        self.trained_model = RandomForestClassifier(max_depth=10)
        self.trained_model.fit(self.x_train, self.y_train)
        self.predictions = self.trained_model.predict(self.x_test)
        print(classification_report(self.predictions, self.y_test))
        save_or_not = input("Save Model? (y or n) ")
        if save_or_not == "y":
            models_count = sum(x.suffix==".pickle" for x in MODELS.iterdir())
            model_name = MODELS/f"rand_forest({models_count}).pickle"
            self.save_model(self.trained_model, model_name)
            print(f"\n\nModel saved as {model_name.name}")
        else:
            print("Model not saved")
        

    def save_model(self, model, model_name):
        with open(model_name, "wb") as f:
            pickle.dump(model, f)

    def load_model(self, model_name=None):
        """
        If not model_name is provided, the latest model will be loaded.

        Args:
            model_name ([type], optional): [description]. Defaults to None.
        """
        if not model_name:
            model_name = str(latest_file())
        with open(model_name, "rb") as f:
            self.trained_model = pickle.load(f)            
            
    def has_malaria(self):
        # Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        out = ["Infected", "Uninfected"]
        filename = askopenfilename(initialdir = str(HOME), title = "Select Malaria cell image", filetypes = (("png files","*.png"),("jpeg files","*.jpg")))
        inp = ImageProcessor().single_image(pathlib.Path(filename))
        
        results = self.trained_model.predict_proba(inp)
        print(results.argmax())
        final = f"\n\nPrediction: {out[results.argmax()]}\nProbability: {results.max()}\n\n"
        return print(final)
        
        
if __name__ == "__main__":
    malaria_model = MalariaModel()
    malaria_model.has_malaria()
    while True:
        cont = input("Predict New? (y or n)")
        if cont!="y":
            break
        malaria_model.has_malaria()
        