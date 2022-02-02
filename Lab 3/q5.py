"""
Created on Wed Feb  2 12:25:39 2022

@author: KIIT
"""

import pandas as pd
import seaborn as sns

df = pd.read_csv('Social_Network_Ads.csv')
#print(df)

sns.set(style='darkgrid')

sns.lmplot(
        x='Age',
        y='EstimatedSalary',
        data=df,
        hue='Purchased',
        fit_reg=False,
        legend=True,
        )
