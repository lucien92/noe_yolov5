import os
import cv2

#On souhaite récupérer tous les crop des images annotées et enregistré chacun des crop dans un dossier portant le nom du label

path_to_images = "/home/lucien/projet_lepinoc/data/data"

def convert_bbox_format_reverse(bbox):
            # bbox est une liste ou un tuple de la forme [x_center, y_center, width, height]
            xmin = int(bbox[0] - 0.5 * bbox[2])
            ymin = int(bbox[1] - 0.5 * bbox[3])
            xmax = int(bbox[0] + 0.5 * bbox[2])
            ymax = int(bbox[1] + 0.5 * bbox[3])
            return xmin, ymin, xmax, ymax

def convert_relativ_to_absolute(bbox, img_width, img_height):
            # bbox est une liste ou un tuple de la forme [xcenter, ycenter, width, height]
            # img_width et img_height sont les dimensions de l'image
            bbox[0] = int(bbox[0] * img_width)
            bbox[1] = int(bbox[1] * img_height)
            bbox[2] = int(bbox[2] * img_width)
            bbox[3] = int(bbox[3] * img_height)
            return bbox

try:
    os.mkdir("/home/lucien/projet_lepinoc/data/crop/")
except:
    pass

for folder in os.listdir(path_to_images):
    folder_path = os.path.join(path_to_images, folder)
    with open(f'{folder_path}/classes.txt') as f:
        lines = f.readlines()
        labels = [line.replace('\n', '') for line in lines]
    folder_ann = os.path.join(folder_path, "ann")
    for annot in os.listdir(folder_ann):
        annot_path = os.path.join(folder_ann, annot)#ceci est le chemin vers le fichier d'annatation
        image_path = annot_path.replace(f'/data/{folder}/ann', '/all_img').replace('.txt', '.jpg')#ceci est le chemin vers l'image correspondante
        with open(annot_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '').split(' ')
                label = labels[int(line[0])]
                img = cv2.imread(image_path)
                bbox = convert_relativ_to_absolute([float(line[1]), float(line[2]), float(line[3]), float(line[4])], img.shape[1], img.shape[0])
                bbox = convert_bbox_format_reverse(bbox)
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[2]
                ymax = bbox[3]
                crop = img[ymin:ymax, xmin:xmax]
                try:
                    os.mkdir(f'/home/lucien/projet_lepinoc/data/crop/{label}') #attention il faut changer les labels à chaque folder car le 1 du dossier 1 n'est pas celui du dossier 4 (pour les 4 grands dossier d'annotations)
                except:
                    pass
                cv2.imwrite(f'/home/lucien/projet_lepinoc/data/crop/{label}/{annot[:-4]}_{label}.jpg', crop)
                print(f'crop saved to /home/lucien/projet_lepinoc/data/crop/{label}/{annot[:-4]}_{label}.jpg')
