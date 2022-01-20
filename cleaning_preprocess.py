import cv2, pathlib, csv
import pandas as pd


HOME = pathlib.Path().resolve().parent /"Malaria"
IMG_DIR = HOME / 'Images'
DATASETS = HOME / 'Datasets'
PARASITISED = IMG_DIR / 'Parasitized'
UNINFECTED = IMG_DIR / 'Uninfected'


class ImageProcessor:
    def __init__(self):
        self.out_df = pd.DataFrame(columns=["Contour_0", "Contour_1", "Contour_2", "Contour_3", "Contour_4"])
        
    def single_image(self, file):
        img = cv2.imread(str(file))
        img = cv2.GaussianBlur(src=img, ksize=(5,5), sigmaX=2)
        gray_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        
        _, threshold = cv2.threshold(src=gray_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(image=threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[:5]
        
        row = {f"Contour_{i}":cv2.contourArea(v) for i, v in enumerate(contours)}
        print(f"{file.stem} processed")
        
        self.out_df = self.out_df.append(row, ignore_index=True)
        return self.out_df.fillna(0)

    def process_dir(self, imgdir=None):
        if imgdir is None:
            return "No directory specified"
        elif not imgdir.is_dir():
            return "Not a directory"
            
        for file in imgdir.iterdir():
            if file.suffix == ".png":
                img = cv2.imread(str(file))
                img = cv2.GaussianBlur(src=img, ksize=(5,5), sigmaX=2)
                gray_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
                
                _, threshold = cv2.threshold(src=gray_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(image=threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[:5]
                
                row = {f"Contour_{i}":cv2.contourArea(v) for i, v in enumerate(contours)}
                row["Label"] = imgdir.name
                print(f"{file.stem} processed - {imgdir.name}")
                
                self.out_df = self.out_df.append(row, ignore_index=True)
        return print("Completed!")
    def process_and_export(self):
        self.process_dir(PARASITISED)
        self.process_dir(UNINFECTED) 
        self.out_df.fillna(0, inplace=True) 
        self.out_df = self.out_df.sample(frac=1)
        self.out_df.to_csv(DATASETS/ "malaria_dataset.csv", index=False)

if __name__ == "__main__":
    processor = ImageProcessor()
    processor.process_and_export()



