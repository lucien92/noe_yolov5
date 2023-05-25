import PIL
from PIL import Image
import csv 
import os

def cropp():
    path_to_data = "/home/lucien/projet_lepinoc/data/test_detect"
    path_to_csv = "/Workspace/Repos/b00786574@essec.edu/noe_yolov5//prediction_results/exp/labels/xyxy"
    output_path = "/Workspace/Repos/b00786574@essec.edu/noe_yolov5//crop/Cropped_Anthophila"

    try:
        os.mkdir("/Workspace/Repos/b00786574@essec.edu/noe_yolov5//crop/Cropped_Anthophila")
    except:
        pass

    def crop_image(img_path, xmin, ymin, xmax, ymax):
        '''
        Crop image from xmin, ymin, xmax, ymaxcoordinates
        '''
        img = PIL.Image.open(img_path)
        img = img.crop((xmin, ymin, xmax, ymax))
        
        return img

    for annotation in os.listdir(path_to_csv):
        if annotation.endswith(".txt"):
            with open(os.path.join(path_to_csv, annotation), "r") as f:
                reader = csv.reader(f, delimiter=" ")
                i = 0
                for row in reader:
                    i += 1
                    #print(row)
                    img_path = f'{path_to_data}/{annotation[:-4]}.jpg'
                    xmin = int(row[1])
                    ymin = int(row[2])
                    xmax = int(row[3])
                    ymax = int(row[4])
                    img = crop_image(img_path, xmin, ymin, xmax, ymax)
                    img.save(os.path.join(output_path, f'{i}{img_path.split("/")[-1]}'))
                    print("Image saved to {}".format(os.path.join(output_path, img_path.split("/")[-1])))