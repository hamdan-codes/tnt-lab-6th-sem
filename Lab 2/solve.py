"""
Created on Wed Jan 19 12:40:43 2022

@author: Chaudhary Hamdan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Social_Network_Ads.csv')

print(df)

plt.scatter(
        df['Age'],
        df['EstimatedSalary']
        )
plt.title('Graph Ques 6')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')

plt.show()