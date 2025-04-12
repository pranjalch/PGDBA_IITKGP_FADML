# -*- coding: utf-8 -*-
"""
Name: Pranjal Chakraborty
Roll No: 24BM6JP41
Project No: DPSVM
Project Title: Diabetes Prediction using Support Vector Machines
"""

"""#PART 3: Generative AI Tool (ChatGPT)"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from cvxopt import matrix, solvers

# Suppress solver output
solvers.options['show_progress'] = False

"""## Data Preprocessing"""

def normalize_data(X_train, X_test):
    # compute mean and std from training set
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Apply normalization
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm

"""## Custom SVM Class"""

# 1a. Lagrangian and KKT Conditions for Modified SVM
# Objective: min 0.5 * ||w||^2 + (C/2) * sum(xi_i^2)
# Constraints: y_i (w^T phi(x_i) - b) >= 1 - xi_i, xi_i >= 0

# Lagrangian:
# L(w, b, xi, alpha, beta) = 0.5||w||^2 + (C/2) * sum(xi_i^2)
# - sum_i alpha_i [y_i(w^T phi(x_i) - b) - 1 + xi_i] - sum_i beta_i * xi_i
#
# KKT Conditions:
# 1. Stationarity:
#    dL/dw = w - sum_i alpha_i y_i phi(x_i) = 0
#    dL/db = sum_i alpha_i y_i = 0
#    dL/dxi = C * xi_i - alpha_i - beta_i = 0
# 2. Primal feasibility:
#    y_i (w^T phi(x_i) - b) >= 1 - xi_i, xi_i >= 0
# 3. Dual feasibility:
#    alpha_i >= 0, beta_i >= 0
# 4. Complementary slackness:
#    alpha_i * [y_i (w^T phi(x_i) - b) - 1 + xi_i] = 0
#    beta_i * xi_i = 0

# 1b. Dual Problem:
# max D(alpha) = sum_i alpha_i - 0.5 * sum_i sum_j alpha_i alpha_j y_i y_j K(x_i, x_j) - (1/2C) sum_i alpha_i^2
# Subject to:
#    sum_i alpha_i y_i = 0, alpha_i >= 0

class CustomSVM:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma=0.1):
        self.C = C
        self.kernel_type = kernel
        self.degree = degree
        self.gamma = gamma

    def kernel(self, A, B):
        # Defines supported kernels: linear, polynomial, and RBF
        if self.kernel_type == 'linear':
            return A @ B.T
        elif self.kernel_type == 'polynomial':
            return (1 + A @ B.T) ** self.degree
        elif self.kernel_type == 'rbf':
            A_sq = np.sum(A ** 2, axis=1).reshape(-1, 1)
            B_sq = np.sum(B ** 2, axis=1).reshape(1, -1)
            dist_matrix = A_sq - 2 * A @ B.T + B_sq
            return np.exp(-self.gamma * dist_matrix)

    def fit(self, X, y):
        n_samples = X.shape[0]

        # Compute kernel matrix (Gram matrix)
        K = self.kernel(X, X)

        # Prepare QP problem matrices
        # P includes kernel interactions + regularization term (modified dual)
        P = matrix(np.outer(y, y) * K + (1/self.C) * np.eye(n_samples))
        q = matrix(-1 * np.ones(n_samples))  # Linear term in dual objective
        G = matrix(-1 * np.eye(n_samples))   # Inequality constraint: alpha_i >= 0
        h = matrix(np.zeros(n_samples))
        A = matrix(y.astype(float), (1, n_samples))  # Equality constraint: sum(alpha_i * y_i) = 0
        b = matrix(0.0)

        # Solve the QP problem to get optimal alpha values
        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(sol['x'])

        # Identify support vectors (non-zero alpha values)
        sv = alpha > 1e-5
        self.alpha = alpha[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]

        # Compute bias term 'b' using KKT conditions
        self.b = np.mean([
            y_k - np.sum(self.alpha * self.sv_y * self.kernel(self.sv_X, x_k.reshape(1, -1)))
            for (x_k, y_k) in zip(self.sv_X, self.sv_y)
        ])

    def project(self, X):
        # Reshape self.alpha and self.sv_y to be column vectors
        return np.sum(self.alpha.reshape(-1, 1) * self.sv_y.reshape(-1, 1) * self.kernel(self.sv_X, X), axis=0) + self.b

    def predict(self, X):
        return np.sign(self.project(X))

"""## Hyperparameter Tuning"""

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, prec, rec, f1

# Grid Search using Train-validation split
def custom_svm_grid_search(X_train, y_train, X_val, y_val, param_grid):
    print(f"\n########## Tuning Custom SVM using Train-validation split ##########\n")

    kernels = param_grid['kernels']
    C_vals = param_grid['C_vals']
    gammas =  param_grid['gammas']
    degrees = param_grid['degrees']

    best_f1 = -np.inf
    best_params = {}
    best_scores = {}

    for kernel in kernels:
        for C in C_vals:
            for gamma in gammas if kernel == 'rbf' else [None]:
                for degree in degrees if kernel == 'polynomial' else [None]:
                    model = CustomSVM(C=C, kernel=kernel, gamma=gamma if gamma else 0.1, degree=degree if degree else 3)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    acc, prec, rec, f1 = evaluate_model(y_val, y_pred)
                    print(f"C={C}, kernel={kernel}, gamma={gamma}, degree={degree} || acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
                    if f1 > best_f1: # Choosing best model based on F1 SCORE
                        best_f1 = f1
                        best_params = {'C': C, 'kernel': kernel, 'gamma': gamma, 'degree': degree}
                        best_scores = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

    return best_params, best_scores

# K-Fold Cross Validation (10 Folds)
def linear_svc_cross_validation(X, y, C_vals, folds=10):
    print(f"\n\n########## Tuning LinearSVC using K-Fold CV ##########\n")

    best_f1 = -np.inf
    best_C = C_vals[0]
    best_scores = {}

    # K-Fold Cross Validation
    kf = KFold(n_splits=folds)

    for C in C_vals:
        prec_list, rec_list, f1_list, acc_list = [], [], [], []

        for train_index, val_index in kf.split(X):
            X_train_cv, X_val_cv = X[train_index], X[val_index]
            y_train_cv, y_val_cv = y[train_index], y[val_index]
            model = LinearSVC(C=C, max_iter=10000, dual=True)
            model.fit(X_train_cv, y_train_cv)
            y_pred_cv = model.predict(X_val_cv)
            acc, prec, rec, f1 = evaluate_model(y_val_cv, y_pred_cv)

            prec_list.append(prec)
            rec_list.append(rec)
            f1_list.append(f1)
            acc_list.append(acc)

        avg_acc = np.mean(acc_list)
        avg_prec = np.mean(prec_list)
        avg_rec = np.mean(rec_list)
        avg_f1 = np.mean(f1_list)

        print(f"C={C} || Avg acc={avg_acc:.4f}, Avg prec={avg_prec:.4f}, Avg rec={avg_rec:.4f}, Avg f1={avg_f1:.4f}")

        if avg_f1 > best_f1: # Choosing best model based on F1 SCORE
            best_f1 = avg_f1
            best_C = C
            best_scores = {'acc': avg_acc, 'prec': avg_prec, 'rec': avg_rec, 'f1': avg_f1}

    return best_C, best_scores

"""## Main function"""
import warnings # LinearSVC Liblinear ConvergenceWarning

def main():
    # Load and preprocess data
    data = pd.read_csv("diabetes.csv")
    print(data.head(), "\n")
    print(data.shape, "\n")

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = np.where(y == 0, -1, 1) # Convert labels to {-1, +1}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize
    X_train, X_test = normalize_data(X_train, X_test)

    # Further split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    # Custom SVM training and Hyperparamater Tuning
    param_grid = {
    'kernels': ['linear', 'rbf', 'polynomial'],
    'C_vals' : [0.1, 0.5, 1, 2, 5, 10],
    'gammas' :  [0.001, 0.005] + list(np.arange(0.01, 0.11, 0.01)) + [0.5, 1],
    'degrees' : [2, 3, 4]
    }

    best_params, best_scores = custom_svm_grid_search(X_train_split, y_train_split, X_val_split, y_val_split, param_grid)
    print("\nBest Hyperparameters for Custom SVM: ", best_params)
    print("Best Scores for Custom SVM: ", best_scores)

    final_model = CustomSVM(**{k: v for k, v in best_params.items() if v is not None})
    final_model.fit(X_train, y_train)
    y_pred_test_custom = final_model.predict(X_test)

    acc, prec, rec, f1 = evaluate_model(y_test, y_pred_test_custom)
    print("\nCustom SVM Final Test Results:")
    print(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n")
    print(classification_report(y_test, y_pred_test_custom))

    # LinearSVC training and cross-validation for best C
    warnings.filterwarnings("ignore") # LinearSVC Liblinear ConvergenceWarning
    C_vals=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20]

    best_C, best_scores = linear_svc_cross_validation(X_train, y_train, C_vals=C_vals)
    print("\nBest Hyperparameters for LinearSVC: C = ", best_C)
    print("Best Scores for LinearSVC: ", best_scores)

    clf = LinearSVC(C=best_C, max_iter=10000, dual=True)
    clf.fit(X_train, y_train)
    y_pred_test_sklearn = clf.predict(X_test)

    acc, prec, rec, f1 = evaluate_model(y_test, y_pred_test_sklearn)
    print("\nSklearn LinearSVC Final Test Results:")
    print(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n")
    print(classification_report(y_test, y_pred_test_sklearn))

if __name__ == "__main__":
    main()