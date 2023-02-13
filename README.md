# Evaluating Fungal Feature Importance in Predicting Life Expectancy for Cancer Patients
This is the repository for DSC180B Section B18-1's Project consisting of Benjamin Sacks, Ethan Chan, and Mark Zheng.
This project is an extension of a study on the classification of cancer types using fungal mycobiome counts which can
be found here: https://www.cell.com/cell/fulltext/S0092-8674(22)01127-8.

This project consists of two main machine learning models based upon the data presented in the previously mentioned
study as well as additional metadata collected about each sample that was not used in prior models. The first is a 
regression model to predict the "days to death" continuous metadata variable measuring when the patient died in days
after their sample was taken. The second is a classification model which aims to distinguish between different cancer
stages(I-IV) as opposed to cancer types in the original study.

To run these models, run the run.py files with their corresponding arguments:

Days to Death Regression: run.py dtd

Cancer Stage Classification: run.py cs

Additionally, to run these models on a smaller test set of data, simply add in the "test" argument:

Use Test Data: run.py dtd test