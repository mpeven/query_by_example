import os
import pandas as pd
from tkinter.filedialog import askopenfilename
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from tkinter import *



### Read in output csv files
experiment_dir = 'experiment2'
experiments = []
for e in range(8):
    csv = "{}/outputs_{:02}.csv".format(experiment_dir, e)
    if os.path.isfile(csv):
        experiments.append(pd.read_csv(csv))
df = pd.concat(experiments)



### Print out the output and columns
print("\nFirst few outputs:")
print(df.head())

print("\nColumns:")
for i, c in enumerate(df.columns):
    print("{} - {}".format(i, c))



### Get the ground truth values
row_idx = int(input("Which row contains ground truth labels: "))
try:
    ground_truth_col = list(df.columns)[row_idx]
except IndexError:
    print('Column {} not found'.format(row_idx))



### Get the predictions
row_idx = int(input("Which row contains predicted labels: "))
try:
    prediction_col = list(df.columns)[row_idx]
except IndexError:
    print('Column {} not found'.format(row_idx))




labels = list(df[ground_truth_col])
guesses = list(df[prediction_col])



''' Build a confusion matrix '''
# Number of classes
classes = sorted(set(list(df[prediction_col].unique()) + list(df[ground_truth_col].unique())))

# Create numpy heatmap 2D array
heatmap = np.zeros([len(classes), len(classes)])
for x, y in zip(guesses, labels):
    heatmap[classes.index(x), classes.index(y)] += 1

# Normalize
# heatmap[:,:] /= np.sum(heatmap, axis=0)

# Load into dataframe
confustion_df = pd.DataFrame(heatmap.astype(int))
confustion_df.index = classes
confustion_df.columns = classes
confustion_df.index.name = "Predictions"
confustion_df.columns.name = "Ground Truth"


### Plot
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(confustion_df, ax=ax, annot=True, fmt="d")
plt.show()
