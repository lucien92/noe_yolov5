path = "/media/lucien/My Passport/projet_abeilles/abeilles-cap500/csv/hierarchie_espèces_bombus.csv"
import pandas

pd = pandas.read_csv(path, sep=",")
print(pd)

#on veut remplacer les chiffres de la colonne numéro du dataframe pd par des chiffre allant de 1 à 70

liste = pd['numero'].tolist()
print(liste)
    
count = 1

for i in range(len(liste)):
    liste[i] = count
    count += 1

pd['numero'] = liste

print(pd)

#on veuttransformer le dataframe en csv

with open('/media/lucien/My Passport/projet_abeilles/abeilles-cap500/csv/hierarchie_espèces_bombus_ok.csv', 'w') as f:
    pd.to_csv(f, sep=',', index=False)
