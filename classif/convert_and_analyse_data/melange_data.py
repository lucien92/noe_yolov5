import os

path_inat = "/media/lucien/My Passport/scrapp_inat_lepido/Pictures" #contient les images de inat    
path_other = "/media/lucien/My Passport/crop" #contient les dossiers des classes extraites dans toutes nos images

try:
    os.mkdir("/media/lucien/My Passport/Pictures_melange")
except:
    pass

#on veut parcourir tous les dossiers et mélanger les images de ceux qui ont le même nom pour els enregistrer toutes dans /media/lucien/My Passport/Pictures_melange/nom
created_folders = []
for folder in os.listdir(path_inat):
    for folder2 in os.listdir(path_other):
        if folder == folder2:
            try:
                os.mkdir("/media/lucien/My Passport/Pictures_melange/"+folder)
                created_folders.append(folder)
            except:
                pass
            for file in os.listdir(path_inat+"/"+folder):
                os.rename(path_inat+"/"+folder+"/"+file, "/media/lucien/My Passport/Pictures_melange/"+folder+"/"+file)#on déplace l'image
            for file in os.listdir(path_other+"/"+folder2):
                os.rename(path_other+"/"+folder2+"/"+file, "/media/lucien/My Passport/Pictures_melange/"+folder+"/"+file)
        
for folder in os.listdir(path_inat):
    if folder not in created_folders:
        try:
            os.mkdir("/media/lucien/My Passport/Pictures_melange/"+folder)
        except:
            pass
        for file in os.listdir(path_inat+"/"+folder):
            os.rename(path_inat+"/"+folder+"/"+file, "/media/lucien/My Passport/Pictures_melange/"+folder+"/"+file)

for folder in os.listdir(path_other):
    if folder not in created_folders:
        try:
            os.mkdir("/media/lucien/My Passport/Pictures_melange/"+folder)
        except:
            pass
        for file in os.listdir(path_other+"/"+folder):
            os.rename(path_other+"/"+folder+"/"+file, "/media/lucien/My Passport/Pictures_melange/"+folder+"/"+file)