"""It is recommended to install the required packages above in your venv or conda.
Run this code before any examples in this notebook."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Other libraries from the skylearn package will be included later.

# Configure dataset from the csv in the repo
df = pd.read_csv("Sample_HousingData.csv")

plt.figure(figsize = (16 , 10))
plt_title = "Effect of Number of Bedrooms on Median House Value"
plt.title(plt_title, size = 15)
plt.xlabel("Number of Bedrooms", 
           loc = "center", 
           fontsize = 15)
plt.ylabel("Median Home Price (Thousands of Dollars)", 
           loc = "center",
            fontsize = 15)

# Create a basic scatter plot of these two data columns
# Note that this follows basic statistics, but the point of ML is that we can use
# multivariable datasets to contribute to the impact of a single variable.
plt.scatter(x = df["RM"], y = df["MEDV"])

# We can also use a line of best fit to show the general relationship of these data
slope, intercept = np.polyfit(df["RM"], df["MEDV"], 1)
x_line = np.arange(df["RM"].min(), df["RM"].max(), 0.1)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line)

plt.show()

"""Demonstration of ML using more than one variable to predict home costs."""

#%pip install scikit-learn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Read the data from the given CSV
df = pd.read_csv("Sample_HousingData.csv")

# We also need to clean it (remove null data)
# The model won't work if we have NaN.
df = df.dropna()
df.head(10)

# The standard notation is to use X for the predictor columns and y for the target
# We can now declare the X and y variables as needed (i.e. all columns but the target)
# Since the last column is the target column, we can index as shown below
X = df.iloc[:, :-1]             # All but the last column
y = df.iloc[:, -1]              # Just the last column

# As mentioned above, we can use the train_test_split() function to split X and y into
# training and test sets.

# Note that the test_size parameter defines the test set size (20% of the data in this case)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Instead of writing our own linear regression model (fun), we can just use the library we
# imported above.
# See the documentation about the regression model. We must configure it after initializing
# the class object.
reg = LinearRegression()
reg.fit(X_train, y_train)       # X is the training data, y is the target values (actual)

# The predict() function takes rows of data and produces the corresponding output with the model
# already created.
y_prediction = reg.predict(X_test)

# Now we can use some statistical analysis to see how far away we are from the real values in the
# test set.
# This uses the mean_squared_error library that we had to import, but you can also use looping
# to compute the squared errors.
rmse = np.sqrt(mean_squared_error(y_test, y_prediction))
print(f'RMSE: {rmse}')

"""
There may be overflow errors. I'm not sure if that is due to the dataset or some configuration
in the library. Regardless the functionality should be fine.
"""