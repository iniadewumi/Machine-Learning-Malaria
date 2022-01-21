import pathlib, random, shutil

HOME = pathlib.Path().resolve().parent /"Malaria"
IMG_DIR = HOME / 'Images'
DATA_DIR = HOME / 'data'
PARASITISED = IMG_DIR / 'Parasitized'
UNINFECTED = IMG_DIR / 'Uninfected'

TRAIN = DATA_DIR / "train"
TEST = DATA_DIR / "validation"

p = list(PARASITISED.glob("*.png"))
u = list(UNINFECTED.glob("*.png"))

random.shuffle(p)
random.shuffle(u)



size = int(len(p)*0.3)

validation_p, train_p = p[:size], p[size:]
validation_u, train_u = u[:size], u[size:]

validation_p_to = [TEST / "Parasitized" / x.name for x in validation_p]
validation_u_to = [TEST / "Uninfected" / x.name for x in validation_u]

train_p_to = [TRAIN / "Parasitized" / x.name for x in train_p]
train_u_to = [TRAIN / "Uninfected"/ x.name for x in train_u]

def copy_data(my_files, to_files):
    for i in range(len(my_files)):
        shutil.copy(my_files[i], to_files[i])

copy_data(validation_p, validation_p_to)
copy_data(validation_u, validation_u_to)
copy_data(train_p, train_p_to)
copy_data(train_u, train_u_to)
# validation_from = validation_p + validation_u
# train_from = train_p + train_u

