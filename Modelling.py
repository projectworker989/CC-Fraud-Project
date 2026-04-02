import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_csv('creditcard.csv')
print("Rows, columns:", df.shape)
print("Fraud cases:", df[df["Class"] == 1].shape[0])

# --- Logistic Regression ---
y = df["Class"].astype(int).to_numpy()
X = df.drop(columns=["Class","Time"])
X_const = sm.add_constant(X)

res = sm.GLM(y, X_const, family=sm.families.Binomial()).fit()
print(res.summary())

# --- Logistic Regression Evaluation ---
probs_logistic = res.predict(X_const)
threshold = 0.2
y_pred_log = (probs_logistic >= threshold).astype(int)

TP = np.sum((y_pred_log == 1) & (y == 1))
FP = np.sum((y_pred_log == 1) & (y == 0))
FN = np.sum((y_pred_log == 0) & (y == 1))
TN = np.sum((y_pred_log == 0) & (y == 0))

print("Logistic Regression:")
print("TP:", TP, "FP:", FP, "FN:", FN, "TN:", TN)
precision = TP/(TP+FP) if (TP+FP)>0 else 0
recall = TP/(TP+FN) if (TP+FN)>0 else 0
f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
print("Precision:", precision, "Recall:", recall, "F1:", f1)

# --- Random Forest ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight={0:1, 1:50},
    n_jobs=-1
)
rf.fit(X_train, y_train)

probs_rf = rf.predict_proba(X_test)[:, 1]
threshold = 0.2
y_pred_rf = (probs_rf >= threshold).astype(int)

TP = np.sum((y_pred_rf == 1) & (y_test == 1))
FP = np.sum((y_pred_rf == 1) & (y_test == 0))
FN = np.sum((y_pred_rf == 0) & (y_test == 1))
TN = np.sum((y_pred_rf == 0) & (y_test == 0))

print("\nRandom Forest:")
print("TP:", TP, "FP:", FP, "FN:", FN, "TN:", TN)
precision = TP/(TP+FP) if (TP+FP)>0 else 0
recall = TP/(TP+FN) if (TP+FN)>0 else 0
f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
print("Precision:", precision, "Recall:", recall, "F1:", f1)

# --- Precision-Recall Curve ---
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, probs_rf)
plt.plot(recall_vals, precision_vals)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Random Forest Precision-Recall Curve")
plt.show()