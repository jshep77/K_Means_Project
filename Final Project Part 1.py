#Final Project Part 1
#Joseph Shepherd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    #name the columns
    col = ["Scn", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Class"]
    #read the data and fill the missing data
    data = pd.read_csv('breast-cancer-wisconsin.data', na_values = '?', names = col)
    #change the NaN data points to the mean of the column
    data = data.fillna(data.mean())
    #find the mean, median, variance, and standard devition for each column
    means = np.around(data[col[1:]].mean(), decimals=1)
    median = np.around(data[col[1:]].median(), decimals=1)
    variance = np.around(data[col[1:]].var(), decimals=1)
    std = np.around(data[col[1:]].std(), decimals=1)
    print(data)
    #use the for loop to print out the data
    for i in range(len(col[1:10])):
        print("Attribute", col[1+i], "--------------")
        print("Mean:", " " * (22-len(str("Mean:"))), means[i])
        print("Median:", " " * (22-len(str("Median:"))), median[i])
        print("Variance:", " " * (22-len(str("Variance:"))), variance[i])
        print("Standard Deviation:", " " * (22-len(str("Standard Deviation:"))), std[i])
        dataset = data[col[1+i]]
        fig = plt.figure()
        sp = fig.add_subplot(1, 1, 1)
        sp.set_title(col[1+i])
        sp.set_xlabel("Value of the Attribute")
        sp.set_ylabel("Number of Data Points")
        sp.hist(dataset, bins=10, color = "blue",edgecolor='black', alpha = 0.5)

main()