import os
import pandas

path_to_data = "/home/lucien/projet_lepinoc/data/crop"

count = {}
for folder in os.listdir(path_to_data):
    nb = len(os.listdir(os.path.join(path_to_data, folder)))
    count[folder] = nb

#on veut convertir ce dict en dataframe

df = pandas.DataFrame.from_dict(count, orient='index', columns=['nb_images'])
print(df)
df.to_csv("/home/lucien/projet_lepinoc/lepinoc-detection/classif/convert_data/nb_image_specie.csv", index=True, header=True, sep=',')