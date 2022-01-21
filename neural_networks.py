from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pathlib


HOME = pathlib.Path().resolve().parent /"Malaria"
IMG_DIR = HOME / 'Images'
DATASETS = HOME / 'Datasets'
PARASITISED = IMG_DIR / 'Parasitized'
UNINFECTED = IMG_DIR / 'Uninfected'
MODELS = HOME / 'Models'

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
image_ = next(img for img in PARASITISED.iterdir() if img.suffix == ".png")

img = load_img(image_)  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for _ in datagen.flow(x, batch_size=1,save_to_dir='NN_Images', save_prefix='infected', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely