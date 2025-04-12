# -*- coding: utf-8 -*-
"""
Name: Pranjal Chakraborty
Roll No: 24BM6JP41
Project No: DPSVM
Project Title: Diabetes Prediction using Support Vector Machines
"""

"""#PART 2: WITH Standard Libraries"""

"""## Data Loading"""

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
def read_data(file="diabetes.csv"):
    try:
      df = pd.read_csv(file)
      print(df.head())
      print("\nShape: ", df.shape)
      return df

    except Exception as e:
      print(f"Error in reading file! : {e}")
      raise

"""## Exploratory Data Analysis"""

# EDA
def eda(df):
  print("\nCheck nulls:")
  print(df.isna().sum())
  print("\nCheck # unique items:")
  print(df.nunique())
  print("\nCheck Outcome values:")
  print(df['Outcome'].unique())

"""## Preprocessing"""

def split_train_test(df, train_size=0.8):
    # Features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convert 0 label to -1 to comply with SVM classification
    y = np.where(y == 0, -1, 1)

    # sklearn's train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42, shuffle=True
    )

    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)

    return X_train, y_train, X_test, y_test

def preprocessing(X_train, X_test):
  # mean and var over train split only
  mean = np.mean(X_train, axis=0)
  std_dev = np.std(X_train, axis=0)

  print("\nBefore Normalization")
  print("Mean: ", mean)
  print("\nStd Dev: ", std_dev)

  # Normalize both train and test set
  X_train_norm = (X_train - mean) / std_dev
  X_test_norm = (X_test - mean) / std_dev

  mean_norm = np.round(np.mean(X_train_norm, axis=0))
  std_dev_norm = np.std(X_train_norm, axis=0)

  print("\nAfter Normalization")
  print("Mean: ", mean_norm)
  print("\nStd Dev: ", std_dev_norm)

  return X_train_norm, X_test_norm

"""## SVM CLASS"""

class CustomSVM:
    def __init__(self, C=1.0, kernel='linear', degree=2, gamma=0.1): # Default: kernel type='linear', polynomial kernel degree=2, rbf gamma=0.1
        self.C = C
        self.kernel_type = kernel
        self.degree = degree
        self.gamma = gamma
        self.alpha = None # Lagrange multipliers
        self.w = None # Weight
        self.b = None # Bias

    # Kernel function
    def kernel(self, x1, x2):
        if self.kernel_type == 'linear':
            return np.dot(x1, x2.T) # xi.xj

        elif self.kernel_type == 'poly':
            # Scaled Kernel to prevent numerical overflow
            return ((1 + np.dot(x1, x2.T)) / x1.shape[1]) ** self.degree # [(1 + xi.xj)/dim]^Q

        elif self.kernel_type == 'rbf':
            X1_sq = np.sum(x1**2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(x2**2, axis=1).reshape(1, -1)
            K = np.exp(-self.gamma * (X1_sq - 2 * np.dot(x1, x2.T) + X2_sq)) # e^(-∥γ(xi - xj∥^2)
            return K

        else:
            raise ValueError("Kernel type not supported")

    # Hessian Matrix
    def compute_hessian(self, X, y):
        K = self.kernel(X, X)
        # H[i][j] = { y[i] * y[j] * K(x[i], x[j]) }
        H = y[:, None] * y[None, :] * K
        return H

    # Solve dual optimization problem using scipy.optimize.minimize!
    def solve_dual(self, H, y):
        n = H.shape[0]
        C = self.C

        # Objective function (minimization)
        def objective(Lambda):
            U = np.ones_like(Lambda)
            obj = 0.5 * Lambda @ H @ Lambda.T + (0.5 / C) * np.sum(Lambda**2) - U @ Lambda.T # min L = 1/2(ΛHΛ') + (1/2C)Λ'Λ - UΛ'
            return obj

        # Jacobian (Gradient of objective wrt alpha)
        def jac(Lambda):
            return H @ Lambda + (1 / C) * Lambda - np.ones_like(Lambda)

        # Equality constraint: sum(alpha_i * y_i) = 0
        constraints = {
            'type': 'eq',
            'fun': lambda Lambda: np.dot(Lambda, y),
            'jac': lambda Lambda: y
        }

        bounds = [(0, self.C) for _ in range(n)] # 0 <= alpha <= C
        Lambda0 = np.zeros(n) # Initial alpha(i)s = 0

        # max_iter kept at10000
        result = minimize(fun=objective, x0=Lambda0, jac=jac, bounds=bounds, constraints=constraints,  options={'maxiter': 10000})

        if not result.success:
            raise ValueError("Optimization failed:", result.message)

        return result.x

    # Weights
    def compute_weights(self, alpha, X, y):
        # W = summation[αi yi xi]
        w = np.sum((alpha * y)[:, None] * X, axis=0)
        return w

    # Bias
    def compute_bias(self, K, alpha, y, X=None):
        support_vectors = np.where((alpha > 1e-4) & (alpha < self.C - 1e-4))[0]

        # Linear Kernel
        if self.kernel_type == 'linear':
            # b = yi- W'X
            b_vals = y[support_vectors] - np.dot(X[support_vectors], self.w)
        # Non-linear Kernels
        else:
            # b = yi - summation[αj yj K(xj. xi)] where i belongs to SV
            b_vals = [(y[i] - np.sum(alpha * y * K[:, i])) for i in support_vectors]
        return np.mean(b_vals) if len(b_vals)>0 else 0.0

    # Training
    def train(self, X, y):
        y = y.flatten()
        self.X_train = X
        self.y_train = y

        H = self.compute_hessian(X, y)
        self.alpha = self.solve_dual(H, y)

        if self.alpha is None:
            raise ValueError("Dual optimization could not be solved using scipy.optimize.minimize!")

        if self.kernel_type == 'linear':
            self.w = self.compute_weights(self.alpha, X, y)
            self.b = self.compute_bias(self.kernel(X, X), self.alpha, y, X)
        else:
            K = self.kernel(X, X)
            self.b = self.compute_bias(K, self.alpha, y)

   # Decision function for classification
    def decision_function(self, X):
        if self.kernel_type == 'linear' and self.w is not None:
            return X @ self.w + self.b
        else:
            K = self.kernel(X, self.X_train)
            return (self.alpha * self.y_train) @ K.T + self.b

    # Prediction
    def predict(self, X):
        fx = self.decision_function(X)
        return np.where(fx >= 0, 1, -1)

"""## Hyperparameter Tuning"""

# Evaluation using sklearn.metrics
def evaluate(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f1, accuracy

def build_model(kernel, C, gamma=None, degree=None):
    return CustomSVM(C=C, kernel=kernel, gamma=gamma, degree=degree)

# KFold CV
def grid_search_svm(X, y, param_grid, k=10):
    best_params = None
    best_f1 = -np.inf
    best_scores = {}

    # Using sklearn.model_selection.KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for C in param_grid['C']:
        for kernel in param_grid['kernel']:
                param_list = [{}]

                # poly kernel has no gamma param and rbf has no degree param
                if kernel == 'poly':
                    param_list = [{'degree': d} for d in param_grid['degree']]
                elif kernel == 'rbf':
                    param_list = [{'gamma': g} for g in param_grid['gamma']]

                for params in param_list:
                    degree = params.get('degree', None)
                    gamma = params.get('gamma', None)

                    prec, rec, f1, acc = [], [], [], [] # record scores for each val set

                    for train_idx, val_idx in kf.split(X):
                        X_train_cv, X_val = X[train_idx], X[val_idx]
                        y_train_cv, y_val = y[train_idx], y[val_idx]

                        model = build_model(kernel, C, gamma=gamma, degree=degree)
                        model.train(X_train_cv, y_train_cv)
                        y_pred = model.predict(X_val)

                        precision, recall, f1_score, accuracy = evaluate(y_val, y_pred)

                        prec.append(precision)
                        rec.append(recall)
                        f1.append(f1_score)
                        acc.append(accuracy)

                    avg_prec = np.mean(prec)
                    avg_rec = np.mean(rec)
                    avg_f1 = np.mean(f1)
                    avg_acc = np.mean(acc)

                    print(f"C={C}, kernel={kernel}, gamma={gamma}, degree={degree} || "
                          f"acc={avg_acc:.4f}, prec={avg_prec:.4f}, rec={avg_rec:.4f}, f1={avg_f1:.4f}")

                    if avg_f1 > best_f1:
                        best_f1 = avg_f1
                        best_params = {'kernel': kernel, 'C': C, 'gamma': gamma, 'degree': degree}
                        best_scores = {'acc': avg_acc, 'prec': avg_prec, 'rec': avg_rec, 'f1': avg_f1}

    return best_params, best_scores

"""## sklearn.svm.LinearSVC"""

from sklearn.svm import LinearSVC
import warnings # To supress LinearSVC Liblinear ConvergenceWarning

# Separate KFold CV function for LinearSVC
def grid_search_linearsvc(X, y, param_grid, k=10):
    best_params = None
    best_f1 = -np.inf
    best_scores = {}

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for C in param_grid['C']:
        prec, rec, f1, acc = [], [], [], []

        for train_idx, val_idx in kf.split(X):
            X_train_cv, X_val = X[train_idx], X[val_idx]
            y_train_cv, y_val = y[train_idx], y[val_idx]

            # sklearn uses labels 0 and 1 for binary classification
            y_train_svc = np.where(y_train_cv == -1, 0, 1)
            y_val_svc = np.where(y_val == -1, 0, 1)

            model = LinearSVC(C=C, max_iter=10000, dual=True)
            model.fit(X_train_cv, y_train_svc)
            y_pred = model.predict(X_val)

            precision, recall, f1_score, accuracy = evaluate(y_val_svc, y_pred)
            prec.append(precision)
            rec.append(recall)
            f1.append(f1_score)
            acc.append(accuracy)

        avg_prec = np.mean(prec)
        avg_rec = np.mean(rec)
        avg_f1 = np.mean(f1)
        avg_acc = np.mean(acc)

        print(f"[LinearSVC] C={C} || acc={avg_acc:.4f}, prec={avg_prec:.4f}, rec={avg_rec:.4f}, f1={avg_f1:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_params = {'C': C}
            best_scores = {'acc': avg_acc, 'prec': avg_prec, 'rec': avg_rec, 'f1': avg_f1}

    return best_params, best_scores

"""## Main function"""

def main():
  # Loading and Exploratory Data Analysis
  print("#################### Loading and Exploratory Data Analysis ####################\n")
  df = read_data(file="diabetes.csv")
  eda(df)

  # Preprocessing
  print("\n\n#################### Preprocessing ####################\n")
  X_train, y_train, X_test, y_test = split_train_test(df, train_size=0.8)
  X_train_norm, X_test_norm = preprocessing(X_train, X_test)

  # Training SVM and Hyperparameter Tuning
  print("\n\n#################### Training SVM and Hyperparameter Tuning ####################\n")
  param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.5, 1, 5, 10],
    'gamma': [0.001, 0.005] + list(np.arange(0.01, 0.11, 0.01)) + [0.5, 1],
    'degree': [2, 3, 4]
  }

  best_params, best_scores = grid_search_svm(X_train_norm, y_train, param_grid, k=10) # Normalized X_train
  print("\nBest params: ", best_params)
  print("Best scores: ",best_scores)

  # Testing and Evaluation
  print("\n\n#################### Testing and Evaluation ####################\n")
  final_model = CustomSVM(**best_params) #kwargs
  final_model.train(X_train_norm, y_train)

  y_pred = final_model.predict(X_test_norm) # Normalized X_test
  precision, recall, f1, acc = evaluate(y_test, y_pred)

  print("Test Accuracy:", acc)
  print("Test Precision:", precision)
  print("Test Recall:", recall)
  print("Test F1 Score:", f1)

  class_rpt = classification_report(y_test, y_pred)
  print("\nClassification Report:\n", class_rpt)

  # LinearSVC
  print("\n\n#################### sklearn.svm LinearSVC ####################\n")
  warnings.filterwarnings("ignore")
  param_grid_svc = {
    'C': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
  }

  best_params_svc, best_scores_svc = grid_search_linearsvc(X_train_norm, y_train, param_grid_svc, k=10)
  print("\nBest params LinearSVC: ", best_params_svc)
  print("Best scores LinearSVC: ",best_scores_svc)

  final_model_svc = LinearSVC(C=best_params_svc['C'], max_iter=20000, dual=True)
  final_model_svc.fit(X_train_norm, y_train) # Normalized X_train

  y_pred = final_model_svc.predict(X_test_norm) # Normalized X_test
  precision, recall, f1, acc = evaluate(y_test, y_pred)

  print("\nSVC Test Accuracy:", acc)
  print("SVC Test Precision:", precision)
  print("SVC Test Recall:", recall)
  print("SVC Test F1 Score:", f1)

  class_rpt = classification_report(y_test, y_pred)
  print("\nClassification Report:\n", class_rpt)

if __name__ == "__main__":
    main()