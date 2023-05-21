import os

import pandas as pd

path = "/home/lucien/projet_lepinoc/data/Pictures_melange"
# try:
#     os.mkdir("/home/lucien/projet_lepinoc/data/cleaned_data")
# except:
#     pass

# for folder in os.listdir(path):
#     if len(os.listdir(path + "/" + folder)) > 30:
#         #on copie le dossier et on l'enregiste dans le dossier cleaned_data/folder
#         os.system('cp -r "' + path + '/' + folder + '" "/home/lucien/projet_lepinoc/data/cleaned_data"')

#maintenant ojn veut changer le csv /home/lucien/projet_lepinoc/lepinoc-detection/classif/convert_and_analyse_data/Liste_especes_IA_Lepi_noc_.csv et le csv 
#/home/lucien/projet_lepinoc/lepinoc-detection/classif/convert_and_analyse_data/especes.csv pour qu'ils ne contiennent plsu que les noms des folder encore présents

#on commence par le csv Liste_especes_IA_Lepi_noc_.csv


df = pd.read_csv("/home/lucien/projet_lepinoc/lepinoc-detection/classif/convert_and_analyse_data/Liste_especes_IA_Lepi_noc_.csv", delimiter = ";")
liste_especes = list(os.listdir("/home/lucien/projet_lepinoc/data/cleaned_data"))


suppr = []
print(df)
df = df[~df.index.duplicated()]
for i in range(len(df)):
    if df["Espece"][i] not in liste_especes:
        #on veut supprimer la ligne i de df
        df = df.drop(i, axis=0)

#on a élmininé 13 espèces qui était dans le csv d'Olivia mais qu'on n'avait pas scrappé pour différentes raisons et qu'il n'y avait pas dans notre bdd

#maintenant on veut changer ce dataframe en csv  /home/lucien/projet_lepinoc/lepinoc-detection/classif/convert_and_analyse_data/new_list.csv

df.to_csv("/home/lucien/projet_lepinoc/lepinoc-detection/classif/convert_and_analyse_data/correct_list.csv", sep=";", index=False)


#On veut créer un nouveau dossier avec les images de ces espèces à partir du dossier Pictures_melange
path = "/home/lucien/projet_lepinoc/data/cleaned_data"
try:
    os.mkdir("/home/lucien/projet_lepinoc/data/cleaned_data_above30")
except:
    pass

for folder in os.listdir(path):
    if folder in list(df["Espece"]):
        #on copie le dossier et on l'enregiste dans le dossier cleaned_data/folder
        os.system('cp -r "' + path + '/' + folder + '" "/home/lucien/projet_lepinoc/data/cleaned_data_above30"')


