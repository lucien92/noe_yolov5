from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import os
import re
import wget

# path à modifier en fonction d'où se trouver le chromedriver
path = '/home/lucien/Documents/final_project_Essec/scrapping/chromedriver' #allow me to use selenium and navigate through the website

#Création dossier pour les saumons : il faudra généraliser pour d'autres familles de poissons
try:
   os.mkdir('/home/lucien/projet_lepinoc/lepinoc-detection/classif/scrapping/Lepidoptera')
except:
    print('dossier déjà existant')


options = Options()
options.add_argument("--window-size=1920,1200")
driver = webdriver.Chrome(options=options, executable_path=path)

# Atteindre la page des espèces de lépidoptères
driver.get('https://www.inaturalist.org/observations?place_id=any&taxon_id=47157&view=species')

time.sleep(3)

os.chdir('/home/lucien/projet_lepinoc/lepinoc-detection/classif/scrapping/Lepidoptera')

### SCROLLER EN BAS DE PAGE POUR ATTEINDRE TOUS LES LIENS
SCROLL_PAUSE_TIME = 2

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

time.sleep(3)

# Créer une liste de listes avec tous les liens des espèces et leurs noms
links = []
for species in driver.find_elements(By.CSS_SELECTOR, "div[ng-repeat='t in taxa']"):

    link = species.find_element(By.CSS_SELECTOR, "div[class='photometa']").find_element(By.CSS_SELECTOR,'a').get_attribute('href') #on récupère les liens des espèces
    name = species.find_element(By.CSS_SELECTOR, "a[class^='display-name']").get_attribute('innerHTML').strip().split('>')[-1] #permet de récupérer le nom de l'espèce
    links.append([link,name])

# On boule sur les 71 espèces de lépidoptère , pour chaque espèce on boucle sur les toutes les pages disponibles et on stock les photos dans un dossier du nom de l'espèce

#on ne souhaite récupérer que les espèces qui nous intéresse

liste = ['Acasis viretata', 'Acleris holmiana', 'Acleris variegana', 'Acontia lucida', 'Acrobasis glaucella', 'Agapeta hamana', 'Agapeta zoegana', 'Agrotis exclamationis', 'Agrotis puta', 'Aleimma loeflingiana', 'Anania hortulata', 'Angerona prunaria', 'Apamea crenata', 'Aphomia sociella', 'Aplocera plagiata', 'Apoda limacodes', 'Archips xylosteana', 'Arctia villica', 'Arctornis L-nigrum', 'Atethmia centrago', 'Biston betularia', 'Callimorpha quadripunctaria', 'Campaea margaritaria', 'Camptogramma bilineata', 'Carcina quercana', 'Celypha striana', 'Chiasmia clathrata', 'Chrysocrambus linetellus', 'Chrysoteuchia culmella', 'Colocasia coryli', 'Cosmia trapezina', 'Crambus lathoniellus', 'Crambus pascuella', 'Craniophora ligustri', 'Cryphia algae', 'Cryphia muralis', 'Cyclophora punctaria', 'Cydalima perspectalis', 'Cydia triangulella', 'Dolicharthria punctalis', 'Dysgonia algira', 'Ecleora solieraria', 'Ecpyrrhorrhoe rubiginalis', 'Eilema caniola', 'Eilema complana', 'Eilema griseola', 'Eilema lurideola', 'Elophila lemnata', 'Emmelia trabealis', 'Endotricha flammealis', 'Ennomos erosaria', 'Epiblema foenella', 'Epirrhoe alternata', 'Epirrhoe galiata', 'Eremobia ochroleuca', 'Ethmia bipunctella', 'Eublemma parva', 'Eucrostes indigenata', 'Eudonia lacustrata', 'Eudonia mercurella', 'Eupithecia centaureata', 'Euproctis chrysorrhoea', 'Euprotis similis', 'Euthrix potatoria', 'Euxoa obelisca', 'Evergestis forficalis', 'Falcaria lacertinaria', 'Gastropacha quercifolia', 'Habrosyne pyritoides', 'Homoeosoma sinuella', 'Hoplodrina octogenaria', 'Horisme radicaria', 'Horisme vitalbata', 'Hypena proboscidalis', 'Hypsopygia costalis', 'Idaea aversata', 'Idaea degeneraria', 'Idaea filicata', 'Idaea fuscovenosa', 'Idaea humiliata', 'Idaea infirmaria', 'Idaea moniliata', 'Idaea ostrinaria', 'Idaea rusticata', 'Idaea subsericeata', 'Laspeyria flexula', 'Laxostege sticticalis', 'Ligdia adustata', 'Lomaspilis marginata', 'Lozotaeniodes formosana', 'Lymantria dispar', 'Malacosoma neustria', 'Mesoligia furuncula', 'Mythimna albipuncta', 'Mythimna vitellina', 'Nomophila noctuella', 'Nycteola revayana', 'Ochropleura plecta', 'Oligia furuncula', 'Oncocera semirubella', 'Orthonama obstipata', 'Ostrinia nubilalis', 'Parapoynx stratiotata', 'Pelurga comitata', 'Peribatodes rhomboidaria', 'Phycita roborella', 'Plutella xylostella', 'Pyrausta aurata', 'Pyrausta despicata', 'Pyrausta purpuralis', 'Pyrausta sanguinalis', 'Rhodometra sacraria', 'Rivula sericealis', 'Scoparia pyralella', 'Scoparia subfusca', 'Scopula ornata', 'Scopula rubiginata', 'Sitochroa verticalis', 'Sphinx ligustri', 'Spilonota ocellana', 'Stauropus fagi', 'Stegania trimaculata', 'Synaphe punctalis', 'Tephronia oranaria', 'Thalpophila matura', 'Thaumetopoea processionea', 'Thisanotia chrysonuchella', 'Thyatira batis', 'Timandra comae', 'Tortrix viridana', 'Triodia sylvina', 'Tyta luctuosa', 'Watsonalla uncinula', 'Xestia c-nigrum', 'Zeiraphera isertana'] #rentrer la liste des noms d'espèces que l'on souhaite récupérer

for specie in links:
    print(specie)
    if specie[1] in liste: 
        print(specie[1] + 'is in list')
        j = 0

        os.mkdir(specie[1])
        os.chdir(specie[1])

        print(specie[1])

        i = 0
        res = True 
        while res :

            time.sleep(3)
            
            i += 1
            
            url = specie[0] + str(i) + '&subview=table' #on fait comme si on avait cliqué sur l'image (qui est une imagette) pour atteindre la page de la photo, qui est en fait le même ilen avec un numéro derrière
            
            driver.get(url)

            time.sleep(3)

            SCROLL_PAUSE_TIME = 2

            # Get scroll height
            last_height = driver.execute_script("return document.body.scrollHeight")

            while True:
                # Scroll down to bottom
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                # Wait to load page
                time.sleep(SCROLL_PAUSE_TIME)

                # Calculate new scroll height and compare with last scroll height
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            time.sleep(3)

            links = []
            try :
                for el in driver.find_elements(By.CSS_SELECTOR,"tr[class='ng-scope']"):
                    num = re.search('\".*\"',el.find_element(By.CSS_SELECTOR, "a[class^='img']").get_attribute('style')).group()[1:-1] #on récupère les liens des photos en étudiant le style css
                    links.append(num)
            
            except :
                print('page finished')
            
            print('hello')
            print(len(links))

            if len(links) == 0 or len(os.listdir('.')) >= 300:
                res = False
                os.chdir('..')
                break
            
            
            for y,u in enumerate(links,j+1):
                
                time.sleep(0.5)
                try : 
                    wget.download(u,str(y)+'.jpeg')
                except :
                    print('something went wrong for image at url n°' + str(y) + 'that is : ' + u)
                    break
                w = y
            
            j = w


driver.quit()