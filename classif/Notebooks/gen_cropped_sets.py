#importations
import os
import cv2
#path to images

base_path_train = "/home/acarlier/code/projet_abeilles/abeilles-cap500/cap500/train"
base_path_evaluate = "/home/acarlier/code/projet_abeilles/abeilles-cap500/cap500/val"
base_path_test = "/home/acarlier/code/projet_abeilles/abeilles-cap500/cap500/test"

path_to_cropped_BD = "/home/acarlier/code/projet_abeilles/abeilles-cap500/Cropped_BD_71"
#Pour le train on cherche dans tous les dossiers qui contiennetn le mot andrena

try:
    os.mkdir(base_path_train + "_cropped_bombus")
except:
    pass

base_path_train_cropped = "/home/acarlier/code/projet_abeilles/abeilles-cap500/cap500/train_cropped_bombus"
a = 0
for folder in os.listdir(base_path_train):
    if "Bombus" in folder:
        a += 1
        if folder != '.DS_Store':
            try:
                os.mkdir(base_path_train_cropped + "/" + folder)
            except:
                continue
            for img in os.listdir(base_path_train + "/" + folder):
                #on veut sauvegarder l'image dans le dossier base_path_train + "_andrena" + "/" + folder
                try:
                    image = cv2.imread(path_to_cropped_BD + "/" + folder + "/" + img)
                    cv2.imwrite(str(base_path_train_cropped) + "/" + str(folder) + "/" + img, image)
                except:
                    pass
    
print("nombre de dosssier bombus:", a)
#Même chose pour l'evaluation

try:
    os.mkdir(base_path_evaluate + "_cropped_bombus")
except:
    pass

base_path_val_cropped = "/home/acarlier/code/projet_abeilles/abeilles-cap500/cap500/val_cropped_bombus"

for folder in os.listdir(base_path_evaluate):
    if "Bombus" in folder:
        if folder != '.DS_Store':
            try:
                os.mkdir(base_path_val_cropped + "/" + folder)
            except:
                continue
            for img in os.listdir(base_path_evaluate + "/" + folder):
                #on veut sauvegarder l'image dans le dossier base_path_train + "_andrena" + "/" + folder
                try:
                    image = cv2.imread(path_to_cropped_BD + "/" + folder + "/" + img)
                    cv2.imwrite(str(base_path_val_cropped) + "/" + str(folder) + "/" + img, image)
                except:
                    pass


#Même chose pour le test
try:
    os.mkdir(base_path_test + "_cropped_bombus")
except:
    pass

base_path_test_cropped = "/home/acarlier/code/projet_abeilles/abeilles-cap500/cap500/test_cropped_bombus"

for folder in os.listdir(base_path_test):
    if "Bombus" in folder:
        if folder != '.DS_Store':
            try:
                os.mkdir(base_path_test_cropped + "/" + folder)
            except:
                continue
            for img in os.listdir(base_path_test + "/" + folder):
                #on veut sauvegarder l'image dans le dossier base_path_train + "_andrena" + "/" + folder
                try:
                    image = cv2.imread(path_to_cropped_BD + "/" + folder + "/" + img)
                    cv2.imwrite(str(base_path_test_cropped) + "/" + str(folder) + "/" + img, image)
                except:
                    pass
