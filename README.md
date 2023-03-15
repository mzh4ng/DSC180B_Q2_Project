# Evaluating Fungal Feature Importance in Predicting Life Expectancy for Cancer Patients
This is the repository for DSC180B Section B18-1's Project consisting of Benjamin Sacks, Ethan Chan, and Mark Zheng.
This project is an extension of a study on the classification of cancer types using fungal mycobiome counts which can
be found here: https://www.cell.com/cell/fulltext/S0092-8674(22)01127-8.

This project consists of two main machine learning models based upon the data presented in the previously mentioned
study as well as additional metadata collected about each sample that was not used in prior models. The first is a 
regression model to predict the "days to death" continuous metadata variable measuring when the patient died in days
after their sample was taken. The second is a classification model which aims to distinguish between different cancer
stages(I-IV) as opposed to cancer types in the original study.


INSTRUCTIONS:
To run these models, run the run.py file with 1 argument, the name of the config file for the desired model. 
Ex. "run.py default-cancer-stage.json".
Additionally, there is a notebook in the path notebooks/run.ipynb that can be used to run this program in 
Jupyter Notebook if desired.

Different models can be selected and run using the config files. Config files are json files in the "config" directory. 
They can be edited to change the parameters of the experiment as well as the type of experiment run. Each experiment
only has 1 config file that it uses to increase the customization of experiments without flooding the folder with 
too many config files.

In each config file, there are 3 subcategories: dataset, preprocessing, and model.
<br /> Dataset specifies information about the raw feature tables including which column is the target variable.
<br /> Preprocessing specifies the parameters of the preprocessing including what transformations to apply to each column. 
    Preprocessing can also be turned off if data is already preprocessed with "do_preprocessing".
<br /> Model specifies the parameters of the model as well as cross validation. These are model specific and will vary
    based upon which type of model is being used.
<br /> Additionally, these are some important keys in the config file:
<br /> experiment_name: Specifies the unique id of the experiment. This is important for separating plots in figures.
<br /> experiment_title: Title of the experiment that will be displayed on the graphs
<br /> experiment_type: internal parameter telling the pipeline which class of model to use (classification or regression)