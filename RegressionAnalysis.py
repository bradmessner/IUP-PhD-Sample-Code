# Author:       Brad D. Messner
# Date:         March 5, 2021
# Topic:        Linear Regression
# Python:		V 3.8

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Open Data File
df = pd.read_csv('/Users/bradmessner/Desktop/agiresult.csv')
x = df['AGI'].to_numpy().reshape((-1, 1))
y = df['Result'].to_numpy()

# Create Model
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
rSquared = model.score(x, y)
print('\nLinear Regression: AGI vs Return Result')
print('=======================================')
print('R Squared / Coefficient of Determination: ', rSquared)
print('Intercept: ', model.intercept_)
print('Slope: ', model.coef_)

# Predictions
yPrediction = model.predict(x)  # make predictions

# Display Scatterplot
plt.scatter(x, y)
plt.plot(x, yPrediction, color='red')
plt.show()
