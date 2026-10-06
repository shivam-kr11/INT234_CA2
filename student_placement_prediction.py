#================ IMPORTING LIBRARIES ================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#================ LOADING DATASET ================
data = pd.read_csv("data/student_placement_train.csv")

#================ BASIC DATASET INFORMATION ================
print(data.head())
print(data.shape)
print(data.columns)

#================ DATASET INFORMATION ================
print(data.info())

#================ STATISTICAL SUMMARY ================
print(data.describe())

#================ CHECKING MISSING VALUES ================
print(data.isnull().sum())

#================ CHECKING DUPLICATE VALUES ================
print(data.duplicated().sum())

#================ CHECKING UNIQUE VALUES ================
print(data.nunique())