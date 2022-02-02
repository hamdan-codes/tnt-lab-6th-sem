"""
Created on Wed Feb  2 12:25:39 2022

@author: KIIT
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Data.csv')

print('A')
sns.countplot(
        x='Country',
        data=df
        )
plt.show()

print('B')
sns.countplot(
        x='Country',
        data=df,
        hue='Purchased'
        )
plt.show()

print('C')
sns.boxplot(
        x='Age',
        y='Country',
        hue='Purchased',
        data=df
        )
plt.show()
