
import pathlib
from cleaning_preprocess import ImageProcessor
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import models
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename



HOME = pathlib.Path().resolve().parent /"Malaria"
IMG_DIR = HOME / 'Images'
DATASETS = HOME / 'Datasets'
PARASITISED = IMG_DIR / 'Parasitized'
UNINFECTED = IMG_DIR / 'Uninfected'
MODELS = HOME / 'Models'

img_width, img_height = 139, 139
batch_size = 16


def latest_file(path=MODELS, model_type="NeuralNetwork"):
    files = [x for x in path.glob("*.h5") if model_type in str(x)]
    return max(files, key=lambda x: x.stat().st_ctime)

class NNMalariaModel:
    def __init__(self):
        pass
        
    def has_malaria(self):
        out = ["Infected", "Uninfected"]
        filename = askopenfilename(initialdir = str(HOME), title = "Select Malaria cell image", filetypes = (("png files","*.png"),("jpeg files","*.jpg")))
        img = load_img(filename)
        
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        inp = test_datagen.flow(x)
        
        
        results = self.model.predict(inp)
        print(results.argmax())
        final = f"\n\nPrediction: {out[results.argmax()]}\nProbability: {results.max()}\n\n"
        return print(final)
    
    
    def load_model(self):
        print("Loading latest Neural Network model...")
        model_name = str(latest_file())
        self.model = models.load_model(model_name)       
            
        
if __name__ == "__main__":
    malaria_model = NNMalariaModel()
    malaria_model.has_malaria()
    while True:
        cont = input("Predict New? (y or n)")
        if cont!="y":
            break
        malaria_model.has_malaria()
        