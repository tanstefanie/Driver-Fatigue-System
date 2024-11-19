# Driver-Fatigue-Project

## Introduction
Singapore, a densely populated city of over six million people, faces an increasing challenge with road safety due to the growing number of vehicles on its roads. The Singapore Police Force reports a worrying rise in accident fatalities, from `104` in 2022 to `131` in 2023 ([source](https://www.police.gov.sg/-/media/D4435F72157942D3B323EE4A507D4CFB.ashx)). Though driver fatigue is not the leading cause of road accidents, it is a significant contributing factor, especially when combined with other risky behaviors, such as driving under the influence of alcohol.

Research highlights the prevalence of sleepiness and fatigue among drivers in Singapore; one study found that `32.9%` of taxi drivers experience sleepiness and fatigue on the job ([source](https://pmc.ncbi.nlm.nih.gov/articles/PMC4350472/)). This underscores the need for measures to help drivers become more aware of their fatigue levels to reduce accident risks.

## Problem Statement
Drivers are often unaware of their fatigue levels, continuing to drive in a compromised state, which increases the risk of accidents. This project aims to develop a driver fatigue detection system that alerts drivers to their fatigue levels in real-time, providing an opportunity to prevent fatigue-related incidents and enhance road safety.

Reference : https://www.channelnewsasia.com/singapore/big-read-rising-traffic-accidents-road-culture-4328841

## Datasets
Majority of our datasets used will be from Kaggle. The possible datasets are as follows:
- https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset


## Directory Structure
```
Driver-Fatigue-System
├── content/
│   ├── model-after-augm.h5
│   ├── model.h5
│   └── model2.h5
|
├── haar cascade files
│   ├── haarcascade_frontalface_alt.xml
│   ├── haarcascade_lefteye_2splits.xml
│   └── haarcascade_righteye_2splits.xml
| 
├── model/
│   ├── transfer_learning.ipynb
│   
```