# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:50:38 2024

@author: Jaekyeong
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)

df = pd.read_csv('titanic.csv')
#ct2 = pd.crosstab(df['Sex'], df['Survived'])
#gender = pd.crosstab(df['Sex'], df['Survived'])
#gender = pd.crosstab(df['Sex'], df['Survived'])
#gender.plot(kind='bar')
#plt.xlabel('Gender')
#plt.ylabel('Survived')
#plt.show()

plt.figure(figsize=(14,10))

sns.set(style='whitegrid')

plt.subplot(3,2,1)
sns.countplot(x='Sex', data=df, palette='Set2')
plt.title('Count of Sex')

plt.subplot(3,2,2)
sns.histplot(df['Age'].dropna(), bins=30, kde=True, color='goldenrod')
plt.title('Distribution of Age')


plt.subplot(3,2,3)
sns.histplot(df['Fare'], bins=30, kde=True, color='green')
plt.title('Distribution of Fare')

plt.subplot(3,2,4)
#sns.countplot(x='SibSp', data=df, palette='Set2')
sns.histplot(df['SibSp'], bins=range(int(df['SibSp'].max()+1)), kde=True, discrete=True)
plt.title('Count of SibSp')

plt.subplot(3,2,5)
#sns.countplot(x='Parch', data=df, palette='Set3')
sns.histplot(df['Parch'], bins=range(int(df['SibSp'].max()+1)), kde=True, discrete=True, color='orchid')
plt.title('Count of Parch')

plt.tight_layout()
plt.show()