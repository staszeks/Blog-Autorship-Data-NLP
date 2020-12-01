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


## install dependencies

Using ``conda``:
```shell
$ conda create -n tidio python=3.8.5
$ conda activate tidio
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
