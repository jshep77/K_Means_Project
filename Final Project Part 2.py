#Final Project Part 2
#Joseph Shepherd
import numpy as np
import pandas as pd
import math
pd.set_option('mode.chained_assignment', None)
def initial(data):
    #create a random number for the index of the two initial centroids
    mu2_index = np.random.randint(0,699)
    mu4_index = np.random.randint(0,699)
    #accounting for the small chance that the two random numbers are the same
    while mu2_index == mu4_index:
        mu4_index = np.random.rantin(0,699)
    #assign the two indexed rows to the first and second centroids 
    mu2 = data.loc[mu2_index]
    mu4 = data.loc[mu4_index]
    centroids= [mu2, mu4, mu2_index, mu4_index]
    return centroids

def assign(centroids, data):
    #find the euclidean distance between both centroids and each point
    #distance between points and centroid 1, or mu2:
    col = ["A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
    calculation_df = data[col]
    for i in range(len(calculation_df)):
        a = (pd.Series(calculation_df.loc[i]) - centroids[0])**2
        distance1 = math.sqrt(a.sum())
        b = (pd.Series(calculation_df.loc[i]) - centroids[1])**2
        distance2 = math.sqrt(b.sum())
        if distance1 > distance2:
            data["Predicted_Class"].loc[i] = 4
        else:
            data["Predicted_Class"].loc[i] = 2
    return data

def recompute(data):
    cluster2 = data[data["Predicted_Class"] == 2]
    mu2 = cluster2.iloc[:, 1:10].mean()
    cluster4 = data[data["Predicted_Class"] == 4]
    mu4 = cluster4.iloc[:, 1:10].mean()
    centroids = [mu2, mu4]
    return centroids

def main():
    #name the columns
    col = ["Scn", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Class", 
"Predicted_Class"]
    display_col = ["Scn", "Class", "Predicted_Class"]
    data = pd.read_csv('breast-cancer-wisconsin.data', na_values = '?', names = col)
    data = data.fillna(data.mean())
    df = data[col[1:10]]
    initial_centroids = initial(df)
    clusters = assign(initial_centroids, data)
    centroids = recompute(clusters)    
    iterations = 0
    mu2 = {}
    mu4 = {}
    for i in range(50):
        clusters = assign(centroids, data)
        centroids = recompute(clusters)
        mu2[i] = (centroids[0])
        mu4[i] = (centroids[1])
        iterations += 1
        if i >= 1:
            if  pd.Series.equals(mu4[i], mu4[i-1]) and pd.Series.equals(mu2[i], mu2[i-1]):
                break
            
    print("Randomly selected row", initial_centroids[2], "for centroid mu_2\n\
nInitial centroid mu_2:")
    print(initial_centroids[0], "\n")
    print("Randomly selected row", initial_centroids[3], "for centroid mu_4\n\
nInitial centroid mu_4:")
    print(initial_centroids[1])
    print()
    print("Program ended after", iterations, "iterations\n")
    print("Final centroid mu_2:\n\n", mu2[i])
    print("\nFinal centroid mu_4:\n\n", mu4[i])
    print("\nFinal cluster assignment: \n\n", clusters[display_col].loc[0:20])
    
main()