"""
Created on Wed Feb  2 12:09:12 2022

@author: Chaudhary Hamdan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data.csv')
df.dropna(axis=0, inplace=True)

index = np.arange(len(df['Country'].unique()))
counts = [3,3,3]
plt.bar(index, counts, color = ['blue', 'red', 'green'])
plt.xticks(index, df.Country.unique())


plt.show()

