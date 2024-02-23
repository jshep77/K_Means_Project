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
        mu4_index = np.random.ranDint(0,699)
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

def validation(clusters):
    error24 = clusters[clusters["Predicted_Class"] == 2]
    error24 = error24[error24["Class"] == 4]
    error24_count = error24.value_counts().sum()
    error42 = clusters[clusters["Predicted_Class"] == 4]
    error42 = error42[error42["Class"] == 2]
    error42_count = error42.value_counts().sum()
    error_all = clusters[clusters["Predicted_Class"] != clusters["Class"]].value_counts().sum()
    pclass2 = clusters[clusters["Predicted_Class"] == 2].value_counts().sum()
    pclass4 = clusters[clusters["Predicted_Class"] == 4].value_counts().sum()
    class_all = len(clusters)
    error_B = (error24_count / pclass2) * 100
    error_M = (error42_count / pclass4) * 100
    error_T = (error_all) / (class_all) * 100
    error_package = [error_B, error_M, error_T, error24, error24_count, error42, error42_count, pclass2, pclass4, class_all, error_all]
    return error_package

def main():
    #name the columns
    col = ["Scn", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Class", "Predicted_Class"]
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
    verify = validation(clusters)
    print("Total Errors:\t\t", np.around(verify[2], decimals=1), "% \n")
    while verify[2] > 50:
        print("Error: The clusters are swapped.\nSwapping Predicted_Class\n")
        change_to_4 = clusters[clusters["Predicted_Class"] == 2]
        change_to_4["Predicted_Class"] = 4
        change_to_2 = clusters[clusters["Predicted_Class"] == 4]
        change_to_2["Predicted_Class"] = 2
        clusters = change_to_2.merge(change_to_4, how = "outer")
        verify = validation(clusters)
    
    print("Data points in Predicted Class 2:", verify[7])
    print("Data points in Predicted Class 4:", verify[8], "\n")
    print("Error data points in Predicted Class 2:\n\n", verify[3], "\n")
    print("Error data points in Predicted Class 4:\n\n", verify[5], "\n")
    print("Number of all data points:\t\t", verify[9], "\n")
    print("Number of error data points:\t", verify[10], "\n")
    print("Error rate for class 2:\t\t\t", np.around(verify[0], decimals=1), "%")
    print("Error rate for class 4:\t\t\t", np.around(verify[1], decimals=1), "%")
    print("Total error rate:\t\t\t\t", np.around(verify[2], decimals=1), "%")
    
main()
