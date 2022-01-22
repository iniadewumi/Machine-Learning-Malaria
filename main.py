from cleaning_preprocess import ImageProcessor
import pathlib
from keras.preprocessing.image import img_to_array, load_img
from keras import models
#from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import pickle


HOME = pathlib.Path().resolve().parent /"Malaria"
IMG_DIR = HOME / 'Images'
DATASETS = HOME / 'Datasets'
PARASITISED = IMG_DIR / 'Parasitized'
UNINFECTED = IMG_DIR / 'Uninfected'
MODELS = HOME / 'Models'




class GeneralClassifier:
    def __init__(self):
        print("\nSince train == False, latest model will be loaded!")
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

    def latest_file(self, model_type, path=MODELS):
        files = [x for x in path.glob("*.pickle") if model_type in str(x)]
        return max(files, key=lambda x: x.stat().st_ctime)

    def load_model(self, model_name=None):
        """
        If not model_name is provided, the latest model will be loaded.

        Args:
            model_name ([type], optional): [description]. Defaults to None.
        """
        model_type = input("\n\nWhat classifier do you want to load? \nChoices are RandomForest, KNN or SVM (Default='RandomForest' because it has the highest accuracy): ")
        print(f"Loading latest {model_type} model...")
        if not model_name:
            model_name = str(self.latest_file(model_type=model_type))
        with open(model_name, "rb") as f:
            self.trained_model = pickle.load(f)            

    
    def has_malaria(self):
        # Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        out = ["Infected", "Uninfected"]
        filename = askopenfilename(initialdir = str(HOME), title = "Select Malaria cell image", filetypes = (("png files","*.png"),("jpeg files","*.jpg")))
        inp = ImageProcessor().single_image(pathlib.Path(filename))
        
        results = self.trained_model.predict_proba(inp)
        final = f"\n\nPrediction: {out[results.argmax()]}\nProbability: {(results.max())*100}%\n\n"
        return print(final)
    

class NeuralNetworkClassifier:
    def __init__(self):
        self.load_model()


    def latest_file(self, model_type="NeuralNetwork", path=MODELS):
        files = [x for x in path.glob("*.h5") if model_type in str(x)]
        return max(files, key=lambda x: x.stat().st_ctime)

    def load_model(self):
        print("Loading latest Neural Network model...")
        model_name = str(self.latest_file())
        self.model = models.load_model(model_name)   
        
        
    def has_malaria(self):
        img_width, img_height = 139, 139
        batch_size = 16

        out = ["Infected", "Uninfected"]
        

        filename = askopenfilename(initialdir = str(HOME), title = "Select Malaria cell image", filetypes = (("png files","*.png"),("jpeg files","*.jpg")))
        img = load_img(filename)        
        img = img.resize((img_width, img_height))
        
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        
        results = self.model.predict(x, batch_size=batch_size)
        final = f"\n\nPrediction: {out[round(results[0][0])]}\nHealthy Probability: {(results[0][0])*100}%\n\n"
        return print(final)
    
    
            
    
class MalariaPredictor:
    def __init__(self, model_type=None):
        model_type = model_type or input(
            "\n\nEnter the model type \nThe Neural Network model is more accurate at prediction\nChoices are: GC = General Classifier,  NNC = Neural Network Classifer: "
        )
        if model_type not in ["NNC", "GC"]:
            raise Exception(f"Model type {model_type} not recognized! Please enter either NNC or GC")

        model_choices = {"NNC": NeuralNetworkClassifier, "GC": GeneralClassifier}
        self.predictor = model_choices[model_type]()
        
        self.predictor.has_malaria()  
        # while True:
        #     cont = input("Predict New? (y or n)")
        #     if cont!="y":
        #         break
        #     self.predictor.has_malaria()      






        
if __name__ == "__main__":
    malaria_model = MalariaPredictor()
