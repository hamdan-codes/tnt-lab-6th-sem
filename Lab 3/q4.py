"""
Created on Wed Feb  2 12:25:39 2022

@author: KIIT
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Social_Network_Ads.csv')
#print(df)

sns.set(style='darkgrid')

print('A')
sns.regplot(
        x=df.Age,
        y=df.EstimatedSalary,
        )
plt.show()

print('B')
sns.regplot(
        x=df.Age,
        y=df.EstimatedSalary,
        fit_reg=False
        )
plt.show()

print('C')
sns.regplot(
        x=df.Age,
        y=df.EstimatedSalary,
        fit_reg=False,
        marker='*'
        )



