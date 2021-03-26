# Author:       Brad D. Messner
# Date:         March 26, 2021
# Topic:        Multi Collinearity
# Python:		V 3.8

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Open the Data Set
data = pd.read_csv('/Users/bradmessner/Desktop/TaxData.csv')

# the independent variables set
X = data[['AGI', 'Status', 'Tax', 'Wages']]

# Create the VIF Data Frame
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns

# Calculate VIF  for Each Feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]

# Output the results
print('\nMulti Collinearity with Certain Tax Return Features')
print('=====================================================')

print(vif_data)
