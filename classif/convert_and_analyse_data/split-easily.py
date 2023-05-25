import splitfolders
import os

try:
    os.mkdir("/content/drive/MyDrive/sets_above30/")
except:
    pass
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("dbfs:/FileStore/tables/data_lepinoc/final_cleaned_data", output="dbfs:/FileStore/tables/data_lepinoc/",
    seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) # default values

# Split val/test with a fixed number of items, e.g. `(100, 100)`, for each set.
# To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
# Set 3 values, e.g. `(300, 100, 100)`, to limit the number of training values.
# splitfolders.fixed("/Workspace/Repos/b00786574@essec.educell_images", output="/media/lucien/My Passport/projet_abeilles/abeilles-cap500/cells_split_2",
#     seed=1337, fixed=(100, 100), oversample=False, group_prefix=None, move=False) # default values
