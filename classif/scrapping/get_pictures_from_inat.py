import pandas as pd 
import sqlite3
import os
import shutil

from download_from_inat import download_from_csv


########## INPUTS ##########

## You must provide an iterable with all the taxon names you want to download the images from
## stored in the df_taxa_in_bdd variable

# path to the csv file with all the taxon name 
csv_path = '/home/lucien/projet_lepinoc/lepinoc-detection/classif/scrapping/especes.csv'
# Load the csv file with all the taxon name
df_taxa_in_bdd = pd.read_csv(csv_path, sep=',')
# Get the taxon names as an iterable
df_taxa_in_bdd = df_taxa_in_bdd['LB_NOM']


# path to the sqlite database
sqlite_path = '/home/lucien/projet_lepinoc/lepinoc-detection/classif/scrapping/inat.db'

# path to the folder where the csv and the images will be saved
output_folder = '/media/lucien/My Passport/scrapp_inat_lepido'
try:
    os.mkdir("/home/lucien/projet_lepinoc/data/inat")
except:
    pass

########## UTILS ##########

def name_to_taxon_id(taxon_name,c):
    
    # Execute the query
    c.execute("select taxon_id from taxa where name= ?", (taxon_name,))

    # Fetch the results
    result = c.fetchone()
    print("look at the result : ", result)

    return result[0]


def info_from_taxon_id(taxon_id,c):
    """
    Return the taxon_id, photo_id, extension and observation_uuid of the taxon_id
    as a dataframe
    """

    # Execute the query pour récupérer les taxon (id dans l'url) qui  nous inétresse
    print(f"on récupère les info du taxon {taxon_id}")
    c.execute("SELECT taxon_id, photo_id, extension, photos.observation_uuid FROM observations INNER JOIN photos on photos.observation_uuid = observations.observation_uuid where taxon_id = ? AND quality_grade = 'research' LIMIT 200", (taxon_id,))

    # Fetch the results
    #result = c.fetchone()
    result = c.fetchall()
    # Create a dataframe with the result
    df = pd.DataFrame(result, columns=['taxon_id', 'photo_id', 'extension', 'observation_uuid'])
    return df


########## MAIN ##########

def main():

    # Connect to the database
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()

    # Creates an output folder to save the csv files
    path_to_csv_files = os.path.join(output_folder,'csv_files')
    if not os.path.exists(path_to_csv_files):
        os.mkdir(path_to_csv_files)

    # Creates an output folder to save the images
    path_to_img_files = os.path.join(output_folder,'Pictures')
    if not os.path.exists(path_to_img_files):
        os.mkdir(path_to_img_files)

    errors = []
   
    # For each taxon name
    for taxon_name in df_taxa_in_bdd:

        print('Downloading the images of the taxon: ' + taxon_name  + '...')

        # Get the taxon id

        try :
            taxon_id = name_to_taxon_id(taxon_name,c)

        except TypeError:

            smiley_error = u'\u274C'
            print('No taxon id for the taxon: ' + taxon_name + '  '+ smiley_error+ '\n')
            errors.append(taxon_name)
            continue

        # Get the info from the taxon id
        df_info_taxon = info_from_taxon_id(taxon_id,c)
        # Save the infos in a csv file
        df_info_taxon.to_csv(path_to_csv_files + '/' + taxon_name + '.csv', index=False)
        # Download the images
        download_from_csv(path_to_csv_files + '/' + taxon_name + '.csv',taxon_name,images_folder=path_to_img_files)

        smiley_done = u'\u2705'
        print('Done for the taxon: ' + taxon_name + '  '+ smiley_done+ '\n')

    print('The following taxon names have not been found in the database: ')
    for error in errors:
        print(error + '\n')

    # Close the connection
    conn.close()

    # Remove the csv files
    shutil.rmtree(path_to_csv_files)

if __name__ == "__main__":
    main()