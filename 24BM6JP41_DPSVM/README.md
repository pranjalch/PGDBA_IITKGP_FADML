# PART 1: fadml_project_dpsvm_p1_(without_libraries).py

## Requirements:
a) Libraries
- import pandas as pd
- import numpy as np

b) File:
- Diabetes.csv

## CustomSVM Class
- Dual Objective Function: min L = 1/2(ΛHΛ') + (1/2C)Λ'Λ - UΛ'
- Dual Optimization Approach: Hessian and Projected Gradient Descent

## Hyperparamater Tuning
- K-Fold Cross Validation (K = 10 Folds)


# PART 2: fadml_project_dpsvm_p2_(with_libraries).py

## Requirements:
a) Libraries
- import pandas as pd
- import numpy as np
- from sklearn.model_selection import train_test_split
- from scipy.optimize import minimize
- from sklearn.model_selection import KFold
- from sklearn.metrics import accuracy_score, classification_report
- from sklearn.metrics import precision_score, recall_score, f1_score
- from sklearn.svm import LinearSVC
- import warnings # !!To supress LinearSVC Liblinear ConvergenceWarning!!

b) File:
- Diabetes.csv

## CustomSVM Class
- Dual Objective Function: min L = 1/2(ΛHΛ') + (1/2C)Λ'Λ - UΛ'
- Dual Optimization Approach: scipy.optimize.minimize

## Hyperparamater Tuning
- K-Fold Cross Validation (K = 10 Folds)


# PART 3: fadml_project_dpsvm_p3_(gen_ai_tool).py

## Requirements:
a) Libraries
- import numpy as np
- import pandas as pd
- from sklearn.model_selection import train_test_split, KFold
- from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
- from sklearn.svm import LinearSVC
- from cvxopt import matrix, solvers
- import warnings # !!To supress LinearSVC Liblinear ConvergenceWarning!!

b) File:
- Diabetes.csv

## CustomSVM Class
- Dual Objective Function: min L = 1/2(ΛHΛ') + (1/2C)Λ'Λ - UΛ'
- Dual Optimization Approach: CVXOPT (matrix, solvers)

## Hyperparamater Tuning
- Grid Search Train-validation split for CustomSVM
- K-Fold Cross Validation (K = 10 Folds) for LinearSVC








