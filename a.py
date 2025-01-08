# %%
import pandas as pd
import numpy as np
import json

# %% [markdown]
# # Load the data

# %%
df_train = pd.read_csv("data/data_train.csv")
df_test = pd.read_csv("data/data_test.csv")

# %%
with open("data/data_ids.json") as f:
    ids = json.load(f)

max_movieId = 0
max_userId = 0
for values in ids["moviesIDs"]:
    max_movieId = max(max_movieId, int(values))
for values in ids["userIDs"]:
    max_userId = max(max_userId, int(values))

# Create a matrix of users and movies
base_matrix = np.zeros((max_movieId + 1, max_userId + 1))
base_matrix.shape

# %%
# Utility matrix and Binary matrix
def utility_matrix_AND_binary_matrix(dfTrain, zerosMatrix = base_matrix, jsonIds = ids):
    Y = zerosMatrix.copy()
    R = zerosMatrix.copy()
    for i, row in dfTrain.iterrows():
        original_movieId, original_userId = int(row["movieId"]), int(row["userId"])
        movieId = jsonIds["moviesIDs"][str(original_movieId)]
        userId = jsonIds["userIDs"][str(original_userId)]
        Y[movieId][userId] = row["rating"]
        R[movieId][userId] = 1
    
    return Y, R

Y, R = utility_matrix_AND_binary_matrix(df_train)

# %%
print(Y)

# %%
print(R)


