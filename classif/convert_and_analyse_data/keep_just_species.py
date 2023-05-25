import os
import pandas as pd
path_to_hierarchy = "/Workspace/Repos/b00786574@essec.edu/noe_yolov5/classif/convert_and_analyse_data/Liste_especes_IA_Lepi_noc_.csv"
path_to_data = "/home/lucien/projet_lepinoc/data/cleaned_data_above30"
try:
    os.mkdir("/home/lucien/projet_lepinoc/data/final_cleaned_data")
except:
    pass

df = pd.read_csv(path_to_hierarchy, delimiter = ";")
espece = list(df["Espece"])

for espece in os.listdir(path_to_data):
    if espece in espece:
        os.system('cp -r "' + path_to_data + '/' + espece + '" "/home/lucien/projet_lepinoc/data/final_cleaned_data"')

#on veut supprimer le dossier path_to_data

os.system("rm -r " + path_to_data)