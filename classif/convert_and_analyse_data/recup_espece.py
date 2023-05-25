import pandas as pd
import os

path_csv = "/Workspace/Repos/b00786574@essec.edu/noe_yolov5/classif/convert_and_analyse_data/correct_list.csv"

#on veut convertir ce csv en dataframe pandas

df = pd.read_csv(path_csv, delimiter = ';')
print(list(df["Espece"]))
print(len(list(df["Espece"])))
L = list(df["Espece"])
#L = ['Acasis viretata', 'Acleris holmiana', 'Acleris variegana', 'Acontia lucida', 'Acrobasis glaucella', 'Agapeta hamana', 'Agapeta zoegana', 'Agrotis exclamationis', 'Agrotis puta', 'Aleimma loeflingiana', 'Anania hortulata', 'Angerona prunaria', 'Apamea crenata', 'Aphomia sociella', 'Aplocera plagiata', 'Apoda limacodes', 'Archips xylosteana', 'Arctia villica', 'Arctornis L-nigrum', 'Atethmia centrago', 'Biston betularia', 'Callimorpha quadripunctaria', 'Campaea margaritaria', 'Camptogramma bilineata', 'Carcina quercana', 'Celypha striana', 'Chiasmia clathrata', 'Chrysocrambus linetellus', 'Chrysoteuchia culmella', 'Colocasia coryli', 'Cosmia trapezina', 'Crambus lathoniellus', 'Crambus pascuella', 'Craniophora ligustri', 'Cryphia algae', 'Cryphia muralis', 'Cyclophora punctaria', 'Cydalima perspectalis', 'Cydia triangulella', 'Dolicharthria punctalis', 'Dysgonia algira', 'Ecleora solieraria', 'Ecpyrrhorrhoe rubiginalis', 'Eilema caniola', 'Eilema complana', 'Eilema griseola', 'Eilema lurideola', 'Elophila lemnata', 'Emmelia trabealis', 'Endotricha flammealis', 'Ennomos erosaria', 'Epiblema foenella', 'Epirrhoe alternata', 'Epirrhoe galiata', 'Eremobia ochroleuca', 'Ethmia bipunctella', 'Eublemma parva', 'Eucrostes indigenata', 'Eudonia lacustrata', 'Eudonia mercurella', 'Eupithecia centaureata', 'Euproctis chrysorrhoea', 'Euprotis similis', 'Euthrix potatoria', 'Euxoa obelisca', 'Evergestis forficalis', 'Falcaria lacertinaria', 'Gastropacha quercifolia', 'Habrosyne pyritoides', 'Homoeosoma sinuella', 'Hoplodrina octogenaria', 'Horisme radicaria', 'Horisme vitalbata', 'Hypena proboscidalis', 'Hypsopygia costalis', 'Idaea aversata', 'Idaea degeneraria', 'Idaea filicata', 'Idaea fuscovenosa', 'Idaea humiliata', 'Idaea infirmaria', 'Idaea moniliata', 'Idaea ostrinaria', 'Idaea rusticata', 'Idaea subsericeata', 'Laspeyria flexula', 'Laxostege sticticalis', 'Ligdia adustata', 'Lomaspilis marginata', 'Lozotaeniodes formosana', 'Lymantria dispar', 'Malacosoma neustria', 'Mesoligia furuncula', 'Mythimna albipuncta', 'Mythimna vitellina', 'Nomophila noctuella', 'Nycteola revayana', 'Ochropleura plecta', 'Oligia furuncula', 'Oncocera semirubella', 'Orthonama obstipata', 'Ostrinia nubilalis', 'Parapoynx stratiotata', 'Pelurga comitata', 'Peribatodes rhomboidaria', 'Phycita roborella', 'Plutella xylostella', 'Pyrausta aurata', 'Pyrausta despicata', 'Pyrausta purpuralis', 'Pyrausta sanguinalis', 'Rhodometra sacraria', 'Rivula sericealis', 'Scoparia pyralella', 'Scoparia subfusca', 'Scopula ornata', 'Scopula rubiginata', 'Sitochroa verticalis', 'Sphinx ligustri', 'Spilonota ocellana', 'Stauropus fagi', 'Stegania trimaculata', 'Synaphe punctalis', 'Tephronia oranaria', 'Thalpophila matura', 'Thaumetopoea processionea', 'Thisanotia chrysonuchella', 'Thyatira batis', 'Timandra comae', 'Tortrix viridana', 'Triodia sylvina', 'Tyta luctuosa', 'Watsonalla uncinula', 'Xestia c-nigrum', 'Zeiraphera isertana']

error = []
with open('/Workspace/Repos/b00786574@essec.edu/noe_yolov5/classif/convert_and_analyse_data/especes.csv', 'w') as f:
    f.write('Especes,0\n')
    for item in L:
        path_to_specie = f'dbfs:/FileStore/tables/data_lepinoc/Pictures_melange/{item}'
        try:
            nb = len(os.listdir(path_to_specie))
            f.write(f'{item},{nb}' + '\n')
        except:
            print(f"error with {item}")
            error.append(item)
            continue
print(error)
