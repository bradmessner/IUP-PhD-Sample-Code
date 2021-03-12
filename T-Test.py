# Author:       Brad D. Messner
# Date:         March 5, 2021
# Topic:        T-Test
# Python:		    V 3.8

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Open Data File
df= pd.read_csv("/Users/bradmessner/Desktop/taxInfo.csv")
df.groupby("status")['agi'].describe()

singleTP = df[(df['status'] == 'single')]
hohTP = df[(df['status'] == 'hoh')]
stats.levene(singleTP['agi'], hohTP['agi'])

singleTP['agi'].plot(kind="hist", title="Single AGI")
plt.xlabel("AGI")
plt.savefig('singleagi_historgram')

hohTP['agi'].plot(kind="hist", title= "Head of Household AGI", color="green")
plt.xlabel("AGI")
plt.savefig('hohagi_histogram')

stats.probplot(singleTP['agi'], dist="norm", plot= plt)
plt.title("Single AGI Q-Q Plot")
plt.savefig("singletp_qqplot.png")

stats.probplot(singleTP['agi'], dist="norm", plot= plt)
plt.title("Head of Household AGI Q-Q Plot")
plt.savefig("hohtp_qqplot.png")

stats.shapiro(singleTP['agi'])
stats.shapiro(hohTP['agi'])
stats.ttest_ind(singleTP['agi'], hohTP['agi'])
