# Blog Autorship Data

## data
Folder for keeping the data and structure for neccessary outputs.

<b> All original data should be kept on private repositories. In order to use the code, data should be copied here. </b>

## notebooks
Folder for jupyter notebooks.
- **EDA** - notebook for explanatory data analysis
- **Training** - notebook for running training pipeline
- **Prediction** - notebook for creating predictions on trained models

<b> All functions developed in notebooks which may be used outside of those notebooks 
should be saved to utilities.</b>

## utilities
Folder with all the shared code as scripts, e.g. helper functions <i>.py</i>.
- utils.py

## standalone files

- requirements.txt
- README.md (this file)
- .gitignore

## load files from OneDrive
In order to speed up the whole process and use already trained models and training history I decided to upload all of the heavy files to OneDrive Storage.

1 x model ~ 400MB x 4 (each task)
raw data ~ 780MB

Please use the link provided below to download all of the neccessary files and copy them into appropriate folder in the repository (each folder name should stay the same).

**https://drive.google.com/drive/folders/1PgSJgz6N2Qw_FtftMZo4ZdZiQ5uuFCOv?usp=sharing**

## install dependencies

Using ``conda``:
```shell
$ conda create -n blog_autorship python=3.8.5
$ conda activate blog_autorship
```

In folder repository:
```shell
$ pip install -r requirements.txt
```
Install kernel for jupyter:
```shell
$ pip install --user ipykernel
$ python -m ipykernel install --user --name=blog_autorship
$ jupyter notebook
```
And we are ready for exploration.
