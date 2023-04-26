import os
import cv2
import pandas as pd


path_to_csv = '/home/lucien/projet_lepinoc/script_lepinoc/script/lepidoptere_detection/src/data/inputs/all_annotations.csv'


def convert_bbox_to_absolute(xmin,ymin,xmax,ymax,width,height):

    """
    Converts the bounding boxes coordinates from relative to absolute.
    If the coordinates are already absolute, nothing is done.
    If the coordinates are negative, equals to 0
    """

    # Check if the coordinates are relative
    if xmin < 1 and xmax < 1 and ymin < 1 and ymax < 1:
            
            # Convert to absolute
            xmin = xmin * width
            xmax = xmax * width
            ymin = ymin * height
            ymax = ymax * height

    # Check if the coordinates are negative
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax < 0:
        xmax = 0
    if ymax < 0:
        ymax = 0

    return xmin,ymin,xmax,ymax,width,height

def draw_bbox_from_csv(path_to_csv,path_to_output=None):
    """
    Draw bounding boxes from a csv file containing the following columns:
    - filepath, xmin, ymin, xmax, ymax, label, width, height
    Creates a new folder of images with bounding boxes drawn on them.
    """

    # Creates the output folder if it doesn't exist
    if path_to_output is None:
        path_to_output = os.path.join(os.path.dirname(path_to_csv), 'with_drawn_bbox')

    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)

    # Load the csv file
    df_dataset = pd.read_csv(path_to_csv, names = ['filepath','xmin','ymin','xmax','ymax','label','width','height'])

    rectangles = []
    # Draw the bounding boxes
    for index, row in df_dataset.iterrows():

        # Load image
        img_path = df_dataset['filepath'][index]
        frame = cv2.imread(img_path)

        # Get the box coordinates
        xmin,ymin,xmax,ymax,width,height = df_dataset['xmin'][index],df_dataset['ymin'][index],df_dataset['xmax'][index],df_dataset['ymax'][index], df_dataset['width'][index],df_dataset['height'][index]
        
        # Draw the box
        line_width_factor = int(min(width, height) * 0.005)

        rectangles.append(((xmin, ymin), (xmax, ymax),(0,255,0)))
        
        if df_dataset['filepath'][index + 1] != img_path:
            for rect in rectangles:
                cv2.rectangle(frame, rect[0], rect[1], rect[2] , line_width_factor * 2)

            # Save image
            img_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(path_to_output,img_name),frame)
            rectangles = []

if __name__ == '__main__':

    draw_bbox_from_csv(path_to_csv)