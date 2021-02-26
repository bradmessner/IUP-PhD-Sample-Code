# Author:       Brad D. Messner
# Date:         February 25, 2021
# Topic:        Nonparametric Statistics
# Python:		V 3.8

import pandas as pd
from scipy.stats import mannwhitneyu
from numpy import mean
import matplotlib.pyplot as plt

# Read data in to a panda from the csv data source
df = pd.read_csv('/Users/bradmessner/Desktop/psByEaCpa.csv')

# Calculate Mean
print('\nGeneral Statistics: EA vs CPA')
print('=============================')
print('Number of Observations: %.0f' % df.shape[0])
print('Mean of EA: %.3f' % (mean(df['EA'].to_numpy())))
print('Mean of CPA: %.3f' % (mean(df['CPA'].to_numpy())))

# Perform Mann-Whitney U Test
print('\nMann-Whitney U Test: EA vs CPA')
print('==============================')
print('Displaying Histogram of EA.')
plt.hist(df['EA'].to_numpy())
plt.show()
print('Displaying Histogram of CPA.')
plt.hist(df['CPA'].to_numpy())
plt.show()
stat, pValue = mannwhitneyu(df['EA'].to_numpy(), df['CPA'].to_numpy())
print('Statistic: %.3f\np Value: =%.3f' % (stat, pValue))
if pValue > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

