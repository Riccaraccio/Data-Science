""" 
This script loads the CHERNAIR dataset and prints the first few rows of the dataset.

dataset from: https://www.kaggle.com/datasets/lsind18/chernobyl-data-air-concentration
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
data = pd.read_csv('ESE-07-Model selection/CHERNAIR.csv', sep=',', header=0)
print(data.columns) # Print the column names of the dataset
print(data.head()) # Print the first few rows of the dataset

# you can acces the data using the following commands:
# data['column_name']
# data['column_name'].values
# data['column_name'].values[0] # first element of the column
# data['column_name'].values[1] # second element of the column

# data.iloc[0] # first row of the data
# data.iloc[1] # second row of the data