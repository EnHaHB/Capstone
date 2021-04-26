# AI decision support for medical triage
## Predicting short and long term outcomes of stroke patients

![](images/stroke_pic.jpeg) 

This is the final project at the Data Science Bootcamp [@neuefische](https://www.neuefische.de/en/weiterbildung/data-science) and has been developed in 4 weeks in the spring of 2021 (by myself).

## Overview

The [International Stroke Trial]() was a randomized, open trial of up to 14 days of antithrombotic therapy started as soon as possible after stroke onset. The aim was to provide reliable evidence on the safety and efficacy of aspirin and of subcutaneous heparin. The dataset was downloaded from here. 
In this project we did not analyse the effect of the treatment participants received during the trail, but used the collected data to train models to predict a negative outcome for the patients (see below). 

We focussed on the investigation whether this data can be implemented to build a decision support tool for medical staff. A ranking process (triage) allows hospitals to prioritize patients based on severity of illness or injury, with the intent of treating the sickest first.


## Goal: Short and long term prediction of stroke patients

The aim of this project was to analyse the data of the International Stroke Trial and its implementation for decision support tools for medical triage. In detail, the objectives were:

+ Build a model that predicts the negative short term outcome, i.e. death after 14 days, of stroke patients.
+ Build a model that predicts the negative long term outcome, i.e. poor health condition or death after six months, of stroke patients.

The prediction of a negative outcome may help medical staff to sort patients into severe and less severe groups and thereby provide the particular care needed.

## Repo Organization

+ [01_IST_clean_feat_eng.ipynb](https://github.com/EnHaHB/Stroke-Outcome/blob/main/01_IST_clean_feat_eng.ipynb): Data cleaning of the original data set. 
+ [02_IST_EDA.ipynb](https://github.com/EnHaHB/Stroke-Outcome/blob/main/02_IST_EDA.ipynb): Exploratory Data Analysis of the data. Includes an introduction on the topic.
+ [03_IST_basic_stats.ipynb](https://github.com/EnHaHB/Stroke-Outcome/blob/main/03_IST_basic_stats.ipynb): Basic simple statistical analyses of the treatment effect participants received during the International Stroke Trial.
+ [04_IST_model_shortterm.ipynb](https://github.com/EnHaHB/Stroke-Outcome/blob/main/04_IST_model_shortterm.ipynb): Analyses and training of various machine learning models to predict the short term outcome of stroke patients.
+ [05_IST_model_longterm.ipynb](https://github.com/EnHaHB/Stroke-Outcome/blob/main/05_IST_model_longterm.ipynb): Analyses and training of various machine learning models to predict the long term outcome of stroke patients.
+ [06_IST_results.ipynb](https://github.com/EnHaHB/Stroke-Outcome/blob/main/06_IST_results.ipynb): Summary of the results, including limitations and thoughts on futur work.


## Outcome



## Conclusions

## Future work

The International Stroke Trial has been continued and also extended in terms of variables. It is worth to look into the dataset. More information can be found [here](https://www.ed.ac.uk/clinical-brain-sciences/research/completed-studies-trials/ist-3-trial).