# Detection of Lepidoptera

Code for the detection of moths (Lepidoptera) 


--------

# Table of contents

1. [Folder contents](#foldercontents)
2. [Conda installation in Linux/Mac](#linuxinstall)
3. [How to Run?](#run)
4. [Output](#results)


---------
## Folder contents <a name="foldercontents"></a>
- data/ folder with test images to be analysed 
- models/ contains trained neural networks, for the detection of moths, or of families of moth:
  - lepidoptera.pt
  
  Model for the detection of lepidoptera, without distinction of family/genus/species.
  - 2families.pt
  
  Model for the detection of 2 families (Crambidae and Geometridae) + Lepidoptera (all other families).
  - 7families.pt
  
  Model for the detection of 7 families (Crambidae, Geometridae, Nolidae, Eberidae, Pyralidae, Noctuidae, Totricidae) + Lepidoptera (all other families).
- utils/ different functions for detection

--------
## 2. Conda installation in Linux/Mac <a name="linuxinstall"></a>

Install anaconda or miniconda (recommended), and create the lepinoc-detection virtual environment by running:
```
conda config --set ssl_verify no
cd /path/to/lepinoc-detection
conda env create --file requirements.yaml
conda activate lepinoc-detection
conda install pytorch==1.7.0 torchvision>=0.8.1 cudatoolkit -c pytorch
```

## 3. How to Run? <a name="run"></a>

To run, activate the conda environment and go to the lepinoc-detection repository folder:
```
conda activate lepinoc-detection

cd /path/to/lepinoc-detection
```
 run the prediction pipeline by executing:
```
python predict.py --weights <path to a pt model> --source <path to images' dir>
```
  
E.g.
```
python predict.py --weights models/2families.pt --source data
```

- for help on deafult parameters and how to modify them, run:

```
python predict.py --help
```

- The results of the prediction will be outputed in `prediction_results` (see [Output](#results))


## 4. Output <a name="results"></a>

Every time the detection software is run, an output folder `prediction_results/exp<n>/` will be generated, together with the following subfolders and file:

  * `img/` : A folder with the images showing the detections

  * `labels/xywh/` : folder containing txt files, such that each line represents a detection for the corresponding image, with the formating:
    class_id box_x box_y box_width box_hight

    *All values are normalized to image size

  * `labels/xyxy/` : folder containing txt files, such that each line represents a detection for the corresponding image, with the formating:
    class_id box_x1 box_y1 box_x2 box_x2

    *Where (x1, y1) is the top left and (x2, y2) the bottom right pixel coordinate of the bounding box representing 
    the detection.
  
  * `detections_summary.csv` : An excel-readable file containing the table of global detections in the dataset, with columns:

    image_name | semantic_class1 | ... | semantic_classN

    *where the semantic classes 1 to N are the detection classes associated to the selected model.


