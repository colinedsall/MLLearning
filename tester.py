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
                + Naive Bayes (Gaussian)
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

""" Algorithm
Main class that defines the set of algorithms a dataset can be learnt from. This class contains
several helper functions to read, write, and maintain the data structure for the relative score
of the data set, depending on if it is best modeled by regressive or classification algos.
"""

class Algorithm:
    """ __init__()
    Initialization function to set up access to the data.

    Parameters:
    + filename:         The name of the file in the current working directory.
    + target_name:      The target column header name. This assumes that columns have names.
    """
    def __init__(self, filename, target):
        try:
            self.df = pd.read_csv(filename, na_values=[" ", "", "NA", "N/A"])
            self.df = self.df.dropna()
            assert not self.df.isnull().values.any(), "Data set still contains NaNs."
        except:
            raise RuntimeError("Cannot clear the NaN data during file parsing.")

        try:
            # If `target` is an integer, use it as column index
            if isinstance(target, int):
                self.y = self.df.iloc[:, target]
                self.X = self.df.drop(self.df.columns[target], axis=1)
            # If `target` is a string, use it as column name
            elif isinstance(target, str):
                if target not in self.df.columns:
                    raise ValueError(f"'{target}' not found in column names.")
                self.y = self.df[target]
                self.X = self.df.drop(columns=[target])
            else:
                raise ValueError("Target must be a column name (str) or column index (int).")

            # Train-test split
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=0)
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
    + AdaBoostRegressor
    + XGBRegressor
    
    + Logistic Regression
    + Naive Bayes (Gaussian)
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
            "AdaBoostRegressor",
            "XGBRegressor",
            "Logistic Regression",
            "Naive Bayes (Gaussian)",
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
    
    """ AdaBoostRegressor()
    Defines the AdaBoostRegressor model from the sklearn library.

    This function returns the mean RMSE.
    """
    def AdaBoostRegressor(self):
        from sklearn.ensemble import AdaBoostRegressor
        return self.regressionModelCV(AdaBoostRegressor())
    
    """ XGBRegressor()
    Defines the XGBRegressor model from the sklearn library.

    This function returns the mean RMSE.
    """
    def XGBRegressor(self):
        from xgboost import XGBRegressor
        return self.regressionModelCV(XGBRegressor())

    """ runRegressors()
    Runs the algorithms as given in order from the array and assigns their rankings to the data
    structure. We can compare the values at another time.
    """
    def runRegressors(self):
        self.rankings.at[0, "Score"] = self.linearRegression()
        self.rankings.at[1, "Score"] = self.KNearestNeighbors()
        self.rankings.at[2, "Score"] = self.decisionTrees()
        self.rankings.at[3, "Score"] = self.randomForest()
        self.rankings.at[4, "Score"] = self.AdaBoostRegressor()
        self.rankings.at[5, "Score"] = self.XGBRegressor()


        # For the remaining algorithms, we can fill the csv with inf for rankings
        for i in range(6, 13, 1):
            self.rankings.at[i, "Score"] = float('inf')

        self.rankings.to_csv("rankings.csv", index = False)
    

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
    Defines the logistic regression model. Note that this is a classification model.

    This function returns the mean score (accuracy).
    """
    def logisticRegression(self):
        from sklearn.linear_model import LogisticRegression

        # Choose maximum iterations at 1000 for efficiency tradeoff
        return self.clfModel(LogisticRegression(max_iter = 1000))
    
    """ GaussianNaiveBayes()
    Defines the function that runs the Gaussian Naive Bayes algorithm.

    This function returns the mean score (accuracy).
    """
    def GaussianNaiveBayes(self):
        from sklearn.naive_bayes import GaussianNB
        return self.clfModel(GaussianNB())
    
    """ KNeighborsClassifier()
    Defines the KNeighborsClassifier function.

    This function returns the mean score (accuracy).
    """
    def KNeighborsClassifier(self):
        from sklearn.neighbors import KNeighborsClassifier
        return self.clfModel(KNeighborsClassifier())
    
    """ DecisionTreeClassifier()
    Defines the DecisionTreeClassifier algo.

    This function returns the mean score (accuracy).
    """
    def DecisionTreeClassifier(self):
        from sklearn.tree import DecisionTreeClassifier
        return self.clfModel(DecisionTreeClassifier(random_state = 0))
    
    """ RandomForestClassifier()
    Defines the RandomForestClassifier algo.

    This function returns the mean score (accuracy).
    """
    def RandomForestClassifier(self):
        from sklearn.ensemble import RandomForestClassifier
        return self.clfModel(RandomForestClassifier(random_state = 0))
    
    """ AdaBoostClassifier()
    Defines the AdaBoost classification algorithm from the sklearn library.

    This function returns the mean score (accuracy).
    """
    def AdaBoostClassifier(self):
        from sklearn.ensemble import AdaBoostClassifier
        return self.clfModel(AdaBoostClassifier())
    
    """ XGBoostClassifier()
    Defines the XGBoost classification algorithm from the xgboost library.

    This function returns the mean score (accuracy).
    """
    def XGBoostClassifier(self):
        from xgboost import XGBClassifier
        return self.clfModel(XGBClassifier())

    """ runClassifiers()
    Runs the classification algorithms in order for the csv output and assigns their rankings to
    the data structure. We can compare these values at another time
    """
    def runClassifiers(self):
        self.rankings.at[6, "Score"] = self.logisticRegression()
        self.rankings.at[7, "Score"] = self.GaussianNaiveBayes()
        self.rankings.at[8, "Score"] = self.KNeighborsClassifier()
        self.rankings.at[9, "Score"] = self.DecisionTreeClassifier()
        self.rankings.at[10, "Score"] = self.RandomForestClassifier()
        self.rankings.at[11, "Score"] = self.AdaBoostClassifier()
        self.rankings.at[12, "Score"] = self.XGBoostClassifier()

        # For the remaining algorithms, we can fill the csv with inf for rankings
        for i in range(0, 6, 1):
            self.rankings.at[i, "Score"] = float('-inf')
    
    """ classificationReport()
    A script to create a classification report for the accuracy of a specific ML model
    on a set of data. This is more efficient than looking at the CV of the sets.
    """
    def classificationReport(self, model):
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        clf = model
        clf.fit(self.X_train, self.y_train)
        y_prediction = clf.predict(self.X_test)
        print(confusion_matrix(self.y_test, y_prediction))
        print(classification_report(self.y_test, y_prediction))
        return clf

    """ determineTargetType(target)
    Function to determine the type of ML algo that the target data must be analyzed by.

    This is required since some libraries' algorithms will throw an exception if the data is
    not suitable for regressive or classification algorithms.

    This function works by comparing the column values as either boolean, enumerated, or
    continuous (floats). Depending on how the column is defined, it will change the respective
    flag for which function to run.
    """
    def determineTargetType(self, target):
        from sklearn.utils.multiclass import type_of_target
        if target.dtype == object or target.dtype.name == 'category':
            self.type = "classification"
        
        # We can also use the sklearn library to determine this too, if the above
        # statement fails
        target_type = type_of_target(target)

        if target_type in ['binary', 'multiclass']:
            self.type = "classification"
        elif target_type in ['continuous', 'continuous-multioutput']:
            self.type = "regression"
        else:
            raise ValueError(f"Unsupported or ambiguous target type: {target_type}")
    
def runAnalysis(algo):
    algo.determineTargetType(algo.y)
    if algo.type == "regression":
        algo.runRegressors()
        algo.rankings = algo.rankings.sort_values(by = algo.rankings.columns[1], ascending = True)
    elif algo.type == "classification":
        algo.runClassifiers()
        algo.rankings = algo.rankings.sort_values(by = algo.rankings.columns[1], ascending = False)

    # Now we should have a set of data that contains the scores (either percentage or RMSE).
    # We can use these values in the datastructure to determine and output the best algo.
    algo.rankings.to_csv("rankings.csv", index = False)
    

""" __main__()
Defines then main function for this Python file. Note that there is some basic CLI if needed
to select the file accessed.
"""
def __main__():
    import argparse
    parser = argparse.ArgumentParser(description="Processes input for ML models.")

    # Optional arguments for CLI
    parser.add_argument("--filename", type = str, help = "Path to the CSV file.",
                        default = "Sample_HousingData.csv")
    parser.add_argument("--target", help = "Target column name or index.",
                        default = "MEDV")

    args = parser.parse_args()

    try:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # Try to interpret target as int index; fallback to string
        try:
            target = int(args.target)
        except ValueError:
            target = args.target

        algo = Algorithm(args.filename, target)
        runAnalysis(algo)

    except:
        raise AttributeError("Incorrect setup.")
    
# Call to main
if __name__ == "__main__":
    __main__()