"""
Filename:       tester.py
Author:         Colin Edsall
Date:           June 3, 2025
Version:        1
Description:    This is a python script that takes an input csv file and validates the most
                accurate (NOT efficient) method to train a ML algorithm for some single-variable
                predictor.

                This script can test quantitative and qualitative predictions using several ML
                algorithms:
                Regressive Algorithms:
                + Linear Regression with Regularization (Ridge)
                + K-Nearest Neighbors
                + Decision Trees
                + Random Forests (generally preferred, hyperparameters default)
                
                Classification Algorithms:
                + Naive Bayes (Gaussian, Multinomial)
                + LogisticRegression (Sigmoid)
                + KNeighborsClassifier
                + DecisionTreeClassifier
                + RandomForestClassifier
                
                Boosting Algorithms
                + AdaBoost
                + XGBoost

                The methodology used to determine the 'most efficient' ML algorithm is the
                root mean squared error (RMSE) if the data is identified to be regressive in
                nature or the weighted average f1-score of from the standard classification report.
"""

import numpy as np                                  # For mathematical analysis and datastructures
import pandas as pd                                 # For I/O and basic operations
import matplotlib.pyplot as plt                     # For data plotting
import warnings                                     # For suppressing warnings

class Algorithm:
    """ __init__()
    Initialization function to set up access to the data.

    Parameters:
    + filename:         The name of the file in the current working directory.
    + target_name:      The target column header name. This assumes that columns have names.
    """
    def __init__(self, filename, target_name):
        try:
            self.df = pd.read_csv(filename, na_values = [" ", "", "NA", "N/A"])
            self.df = self.df.dropna()

            # To verify that the NaN values are dropped
            assert not self.df.isnull().values.any(), "Data set still contains NaNs."
        except:
            raise RuntimeError("Cannot clear the NaN data during file parsing.")

        try:
            # Assign the predictor and target columns from exact-match of the column name
            self.y = self.df[target_name]
            self.X = self.df.drop(columns=[target_name])

            # Create a train test split (80/20% by default) to train all the algorithms with these
            # data.
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                                                    self.y,
                                                                                    test_size = 0.2,
                                                                                    random_state = 0)
        except:
            raise RuntimeError("Cannot create training and testing set.")
        
        """
        We now have our training and testing set for these algorithms. Note that depending on the
        kind of algorithm, these sets will be used differently. Refer to the structure of the
        tested algorithms and their respective source code.
        """

        self.configureAlgorithms()


    """ configureAlgorithms()
    Configures the algorithms in the class array to be tested.

    The list of algorithms is given in the header, but they are as follows;
    + Linear Regression with CV and Ridge Regularization
    + K-Nearest Neighbors with default hyperparameters
    + Decision Trees
    + Random Forests (500 estimators)
    + Logistic Regression
    + AdaBoostRegressor
    + XGBRegressor
    
    + Naive Bayes (Gaussian, Multinomial)
    + KNeighborsClassifier
    + DecisionTreeClassifier
    + RandomForestClassifier
    + AdaBoostClassifier
    + XGBClassifier
    """
    def configureAlgorithms(self):
        self.algos = [
            "Linear Regression with CV and Ridge Regularization",
            "K-Nearest Neighbors",
            "Decision Trees",
            "Random Forests",
            "Logistic Regression",
            "AdaBoostRegressor",
            "XGBRegressor",
            "Naive Bayes (Gaussian)",
            "Naive Bayes (Multinomial)",
            "KNeighborsClassifier",
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "AdaBoostClassifier",
            "XGBClassifier"
        ]

        # Create a list to hold the type of ML algo and the 'score' it receives.
        self.rankings = pd.DataFrame({
            "Algorithm": self.algos,
        })

    """ regressionModelCV(model, k)
    CV optimization for regression models using the negative mean squared error. k is configurable.

    Parameters:
    + model:            The specific model that is being tested, where cross-validation can occur.
    + k:                The number of folds, defaulted to 5.
    """
    def regressionModelCV(self, model, k = 5):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, self.X, self.y, scoring = "neg_mean_squared_error", cv = k)
        rmse = np.sqrt(-scores)

        return rmse.mean()

    """ linearRegression()
    Defines the linear regression model as given, with default k = 5.

    This function returns the mean RMSE.
    """
    def linearRegression(self):
        # For minimization of linear regression error using several folds instead of one (default 5)
        from sklearn.linear_model import LinearRegression

        # Create the Linear Regression algorithm
        self.reg = LinearRegression()
        self.reg.fit(self.X_train, self.y_train)
        
        from sklearn.linear_model import Ridge
        return self.regressionModelCV(Ridge())

    """ KNearestNeighbors()
    K-Nearest Neighbors ML algorithm.

    This function returns the mean RMSE.
    """
    def KNearestNeighbors(self):
        from sklearn.neighbors import KNeighborsRegressor
        return self.regressionModelCV(KNeighborsRegressor()) 
    
    """ decisionTrees()
    Defines the decision tree model.

    This function returns the mean RMSE.
    """
    def decisionTrees(self):
        from sklearn import tree
        return self.regressionModelCV(tree.DecisionTreeRegressor(random_state = 0))
    
    """ randomForest()
    Defines the random forest tree model.

    This function returns the mean RMSE.
    """
    def randomForest(self):
        from sklearn.ensemble import RandomForestRegressor
        return self.regressionModelCV(RandomForestRegressor(random_state = 0, n_jobs = -1,
                                      n_estimators = 500))
    
    """ clfModel(model)
    Takes a model as the input and computes the cross value score for percentages.

    This returns the mean score of the model.
    """
    def clfModel(self, model):
        from sklearn.model_selection import cross_val_score
        clf = model
        scores = cross_val_score(clf, self.X, self.y)
        return scores.mean()

    """ logisticRegression()
    Defines the logistic regression model.

    This function returns the mean score (accuracy).
    """
    def logisticRegression(self):
        from sklearn.linear_model import LogisticRegression

        return self.clfModel(LogisticRegression(max_iter = 1000))

    """ runAlgos()
    Runs the algorithms as given in order from the array and assigns their rankings to the data
    structure. We can compare the values at another time.
    """
    def runAlgos(self):
        self.rankings.at[0, "Score"] = self.linearRegression()
        self.rankings.at[1, "Score"] = self.KNearestNeighbors()
        self.rankings.at[2, "Score"] = self.decisionTrees()
        self.rankings.at[3, "Score"] = self.randomForest()
        # self.rankings.at[4, "Score"] = self.logisticRegression()
        self.rankings.to_csv("rankings.csv", index = False)
    
    

""" __main__()
"""
def __main__():
    import argparse

    try:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        test = Algorithm("Sample_HousingData.csv", "MEDV")
        test.runAlgos()

    except:
        raise AttributeError("Incorrect setup.")
    

# Call to main
if __name__ == "__main__":
    __main__()