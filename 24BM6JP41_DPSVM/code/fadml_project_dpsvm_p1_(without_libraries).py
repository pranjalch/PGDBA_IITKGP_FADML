# -*- coding: utf-8 -*-
"""
Name: Pranjal Chakraborty
Roll No: 24BM6JP41
Project No: DPSVM
Project Title: Diabetes Prediction using Support Vector Machines
"""

"""# PART 1: WITHOUT Any Standard Libraries"""

"""## Data Loading"""

# import libraries
import pandas as pd
import numpy as np

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
  np.random.seed(42)

  data = df.values
  np.random.shuffle(data) # Random shuffling

  # 80-20 split
  split_index = int(train_size * len(data))
  train_data, test_data = data[:split_index], data[split_index:]

  # Features and labels
  X_train, y_train = train_data[:, :-1], train_data[:, -1]
  X_test, y_test = test_data[:, :-1], test_data[:, -1]

  # Convert 0 label to -1 to comply with SVM classification
  y_train = np.where(y_train == 0, -1, 1)
  y_test = np.where(y_test == 0, -1, 1)

  print('X_train: ', X_train.shape)
  print('y_train: ', y_train.shape)
  print('X_test: ', X_test.shape)
  print('y_test: ', y_test.shape)

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

    # Setting up Kernel function
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

    # Gradient of dual objective function
    def dual_obj_gradient(self, Lambda, H, C):
        # Lambda: vector of alpha(i)s, # U: vector of ones
        U = np.ones_like(Lambda)
        # obj = 0.5 * Lambda @ H @ Lambda.T + (0.5 / C) * np.sum(Lambda**2) - U @ Lambda.T # min L = 1/2(ΛHΛ') + (1/2C)Λ'Λ - UΛ'
        grad = H @ Lambda.T + (1 / C) * Lambda.T - U.T # HΛ' + (1/C)Λ' - U'
        return grad

    # Satisfy constraints
    def project_constraint(self, Lambda, y):
        # Project onto the constraints: (1) Lambda >= 0 ; (2) summation(alpha(i) * yi) = 0
        eps = 1e-12  # numerical stability
        Lambda = np.maximum(Lambda, 0)  # Lambda >= 0
        Lambda = Lambda - (y * (np.dot(Lambda, y) / (np.dot(y, y) + eps)))  # Project Lambda onto a hyperplane with normal vector y
        return Lambda

    # Projected gradient descent
    def solve_dual(self, H, y, C, lr=0.001, max_iter=50000):
        n = H.shape[0]
        Lambda = np.zeros((1, n))  # dim: 1 x n

        for _ in range(max_iter):
            grad = self.dual_obj_gradient(Lambda, H, C)
            Lambda = Lambda - lr * grad.T  # Update Lambda
            Lambda = np.clip(Lambda, 0, 1e6) # Added for numerical stability
            Lambda = self.project_constraint(Lambda.flatten(), y).reshape(1, -1)

        return Lambda.flatten()

    # Weights
    def compute_weights(self, alpha, X, y):
        # W = summation[αi yi xi]
        w = np.sum((alpha * y)[:, None] * X, axis=0)
        return w

    # Bias
    def compute_bias(self, K, alpha, y, X=None):
        support_vectors = np.where((alpha > 1e-4) & (alpha < self.C - 1e-4))[0]
        b_vals = []

        # Linear Kernel
        if self.kernel_type == 'linear':
            # b = yi- W'X
            b_vals = y[support_vectors] - np.dot(X[support_vectors], self.w)
            return np.mean(b_vals) if len(b_vals) > 0 else 0.0
        # Non-linear Kernels
        else:
            # b = yi - summation[αj yj K(xj. xi)] where i belongs to SV
            for i in support_vectors:
                w_dash = np.sum(alpha * y * K[:, i])
                b_vals.append(y[i] - w_dash)
            return np.mean(b_vals) if b_vals else 0.0

    # Training
    def train(self, X, y, lr=0.001, max_iter=10000):
        y = y.flatten()
        self.X_train = X
        self.y_train = y

        H = self.compute_hessian(X, y)
        self.alpha = self.solve_dual(H, y, self.C, lr=lr, max_iter=max_iter)
        # print("alpha vector dimension:\n", self.alpha.shape)

        if self.kernel_type == 'linear':
            self.w = self.compute_weights(self.alpha, X, y)
            self.b = self.compute_bias(self.kernel(X, X), self.alpha, y, X)
            # print("Weights:\n", self.w)
        else:
            K = self.kernel(X, X)
            self.b = self.compute_bias(K, self.alpha, y)
        # print("Bias:\n", self.b)

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

def evaluate(y_true, y_pred):
    eps = 1e-8 # To avoid 0 division
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_score = 2 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return precision, recall, f1_score, accuracy

def k_fold_split(X, y, k=10):
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices) # random split
    fold_size = len(X) // k
    folds = []

    for i in range(k):
        val_idx = indices[i * fold_size:(i + 1) * fold_size] # ith data chunk
        train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]]) # Keep data before val set and after val set
        folds.append((X[train_idx], y[train_idx], X[val_idx], y[val_idx]))

    return folds

def build_model(kernel, C, gamma=None, degree=None):
    if kernel == 'poly':
        return CustomSVM(C=C, kernel=kernel, degree=degree)
    elif kernel == 'rbf':
        return CustomSVM(C=C, kernel=kernel, gamma=gamma)
    else:
        return CustomSVM(C=C, kernel=kernel)

def grid_search_svm(X, y, param_grid, k=10):
    best_params = None
    best_scores = None
    best_f1 = -np.inf # CV based on F1 Score
    folds = k_fold_split(X, y, k=k)

    for C in param_grid['C']:
        for kernel in param_grid['kernel']:
            for lr in param_grid['lr']:
                param_list = [{}]

                if kernel == 'poly':
                    param_list = [{'degree': d} for d in param_grid['degree']]
                elif kernel == 'rbf':
                    param_list = [{'gamma': g} for g in param_grid['gamma']]

                for params in param_list:
                    degree = params.get('degree', None)
                    gamma = params.get('gamma', None)

                    prec, rec, f1, acc = [], [], [], [] # record scores for each val set

                    for X_train_cv, y_train_cv, X_val, y_val in folds:
                        model = build_model(kernel, C, gamma=gamma, degree=degree)
                        model.train(X_train_cv, y_train_cv, lr=lr)
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

                    print(f"C={C}, kernel={kernel}, lr={lr}, gamma={gamma}, degree={degree}"
                          f" || acc={avg_acc:.4f}, prec={avg_prec:.4f}, rec={avg_rec:.4f}, f1={avg_f1:.4f}")

                    if avg_f1 > best_f1:
                        best_f1 = avg_f1
                        best_params = {'kernel': kernel, 'C': C, 'gamma': gamma, 'degree': degree, 'lr': lr}
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
    'degree': [2, 3, 4],
    'lr': [1e-3, 5e-3, 1e-4, 5e-4]
  }

  best_params, best_scores = grid_search_svm(X_train_norm, y_train, param_grid, k=10)
  print("\nBest params: ", best_params)
  print("Best scores: ",best_scores)
  best_params.pop('lr', None)

  # Testing and Evaluation
  print("\n\n#################### Testing and Evaluation ####################\n")
  final_model = CustomSVM(**best_params)
  final_model.train(X_train_norm, y_train)

  y_pred = final_model.predict(X_test_norm)
  precision, recall, f1, acc = evaluate(y_test, y_pred)

  print("Test Accuracy:", acc)
  print("Test Precision:", precision)
  print("Test Recall:", recall)
  print("Test F1 Score:", f1)

if __name__ == "__main__":
    main()