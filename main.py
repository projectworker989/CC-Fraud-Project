import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from IPython.display import display as dp


df = pd.read_csv('creditcard.csv')
#pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',500)
print("There are (rows,columns) =", df.shape)
print("The number of fraud cases in this data set is ",df[df["Class"] == 1].shape[0])

dp(df.describe())
dp(df[df["Class"] == 1])


#LOGISTIC REGRESSION USING statsmodels.api 


def logistic(z):
    return 1 / (1 + np.exp(-z))


y = df["Class"].astype(int).to_numpy()          # (n,)
X1 = df.drop(columns=["Class","Time"]).to_numpy()       # (n, p)
X2 = sm.add_constant(X1)                     # add intercept
res = sm.GLM(y, X2, family=sm.families.Binomial()).fit()
beta = res.params

print(beta)

#ITERATIVE REWEIGHTED LEAST SQUARES
def IRLS(df):
    if "Class" not in df.columns:
        raise ValueError("Need 'Class' column to fit the model.")

    y = df["Class"].astype(int).to_numpy() #THe column that you're trying to predict on
    X = df.drop(columns=["Class", "Time"], errors="ignore")  #The features you're trying to predict with
    X = sm.add_constant(X, has_constant="add")#The features as in X1 except we just add a constant.

    res = sm.GLM(y, X, family=sm.families.Binomial()).fit()# code to apply IRLS
    return res.params # an array of the beta values starting from 0 where beta[0] is the itnercept term and each subsequent one corresponds to the other features.



def predict_prob(dataframe, beta, threshold):
    X1 = dataframe.drop(columns=["Class"]).to_numpy()
    X2 = sm.add_constant(X1)
    probabilities = logistic(X2 @ beta)
    y_pred = (probabilities >= threshold).astype(int)
    
    return y_pred


#CROSS VALIDATION FOR TUNING THE THRESHOLD FOR LOGISTIC REGRESSION
#DATAFRAME TO ACTUALLY PREDICT WITH. WE NEED TO TAKE INTO ACCOUNT THE CONSTANT. SO THATS WHY THERES A CONSTANT COLUMN WITH ALL 1's
X5 = df.drop(columns=["Class", "Time"])
X5.insert(0, "constant", 1)
X5 = X5.reset_index(drop=True)
X5.head()

#DATAFRAME TO ACTUALLY POST RESULTS ON PREDICTION WITH
X6 = df.drop(columns=["Time"])
X6["Predicted"]=0
X6 = X6.reset_index(drop=True)


n = len(X6) #size of dataset
rng = np.random.default_rng(67)          # seed for reproducibility
index = np.arange(n) #creates an array of size n called index, storing integers from 0 to n-1 , inclusive.
rng.shuffle(index) #randomly mixes up the array.
Tenfolds = np.array_split(index, 10) # Creates an array of size 10 where each entry is an array of random indices, all of nearly equal size.

thresholds = np.arange(0.05, 1.01, 0.05)
results = []

for Lambda in thresholds:
    print(Lambda)
    mse_folds = []
    for i in range(10):
        #splits into training indices and test indices
        test_index = Tenfolds[i]
        train_index = np.setdiff1d(np.arange(n), test_index)

        df_train = X6.iloc[train_index]
        df_test  = X6.iloc[test_index]
    
        beta=IRLS(df_train)
        y_test = df_test["Class"].astype(int).to_numpy()
        y_hat = predict_prob(df_test, beta,Lambda)
        mse = np.mean((y_test - y_hat) ** 2)
        mse_folds.append(mse)
    
    
    cv_mse = np.mean(mse_folds)
    results.append((Lambda,cv_mse))
    
best = min(results, key=lambda t: t[1])
print("best threshold, best CV MSE:", best)
print(results)



#TESTING AND COMPUTING TRUE POSITIVE, TRUE NEGATIVE, FALSE POSITIVE, FALSE NEGATIVE, AREA UNDER THE ROC (RECEIVER OPERATOR CURVE)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


#DATAFRAME TO ACTUALLY PREDICT WITH. WE NEED TO TAKE INTO ACCOUNT THE CONSTANT. SO THATS WHY THERES A CONSTANT COLUMN WITH ALL 1's
X5 = df.drop(columns=["Class", "Time"])
X5.insert(0, "constant", 1)
X5 = X5.reset_index(drop=True)
print(len(X5.columns))

#DATAFRAME TO ACTUALLY POST RESULTS ON PREDICTION WITH
X6 = df.drop(columns=["Time"])
X6["Predicted"]=0
X6 = X6.reset_index(drop=True)

#Predict with threshold equal to 0.1
for i in range(len(X6)):
    if logistic(X5.iloc[[i]].to_numpy().ravel() @ beta )>=0.1:
        X6.at[X6.index[i],"Predicted"]=1
    else: 
        X6.at[X6.index[i],"Predicted"]=0

count=0
for i in range(0,len(X6)):
    if X6.at[X6.index[i], "Class"] == X6.at[X6.index[i],"Predicted"]:
        count+=1
print("How many incorrectly predicted: ",len(X6)-count)
print("Portion correctly predicted: ",count/len(X6))


#Various Metrics
True_Positive=0
True_Negative=0
False_Positive=0
False_Negative=0

for row_index in range(len(X6)):
    if X6.loc[row_index, "Class"]==1 and X6.loc[row_index, "Predicted"]==1:
        True_Positive+=1
    elif X6.loc[row_index, "Class"]==0 and X6.loc[row_index, "Predicted"]==0:
        True_Negative+=1
    elif X6.loc[row_index, "Class"]==0 and X6.loc[row_index, "Predicted"]==1:
        False_Positive+=1  
    elif X6.loc[row_index, "Class"]==1 and X6.loc[row_index, "Predicted"]==0:
        False_Negative+=1

Accuracy = (True_Positive+True_Negative)/(True_Positive+True_Negative+False_Positive+False_Negative)
Error_Rate = 1-Accuracy
Precision = True_Positive/(True_Positive+False_Positive)
Sensitivity = True_Positive/(True_Positive+False_Negative)
Negative_Predictive_Value = True_Negative/(False_Negative+True_Negative)
Specificity = True_Negative/(False_Positive+True_Negative)
F_1_Score = 2*True_Positive/(2*True_Positive+False_Positive+False_Negative)

print("Size of data set is ", len(X6))
print("True Positive amount is ", True_Positive)
print("True Negative amount is ", True_Negative)
print("False Positive amount is ", False_Positive)
print("False Negative amount is " ,False_Negative)
print("Accuracy is ",Accuracy)
print("Error Rate is ", Error_Rate)
print("Precision is ", Precision)
print("Sensitivity is ", Sensitivity)
print("Negative Predictive Value is ", Negative_Predictive_Value)
print("Specificity is ", Specificity)
print("F1 Score is ", F_1_Score)

#Area Under Curve for LOGISTIC REGRESSION
y_score0 = np.array([
    logistic(X5.iloc[i].to_numpy().ravel() @ beta)
    for i in range(len(X5))
])

y_true0 = X6["Class"].to_numpy()
auc = roc_auc_score(y_true0, y_score0)
print("AUC:", auc)

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X5, X6["Class"], test_size=0.3, random_state=42
)
y_score = np.array([
    logistic(X_test.iloc[i].to_numpy().ravel() @ beta)
    for i in range(len(X_test))
])
auc1 = roc_auc_score(y_test, y_score)
print("Test AUC:", auc1)



#I MADE THINGS INTO FUNCTIONS TO EASILY CALL WHAT WE FREQUENTLY USED.

def UndersampleDataFrame(dataframe,NumberOfClassZeroCopies,SEED_which_must_be_integer):
    # Separate classes
    df_pos = dataframe[dataframe["Class"] == 1]
    df_neg = dataframe[dataframe["Class"] == 0]
    # Randomly sample 1000 negatives
    df_neg_sample = df_neg.sample(n=NumberOfClassZeroCopies, random_state=SEED_which_must_be_integer)
    # Combine
    df_balanced = pd.concat([df_pos, df_neg_sample])
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced


def Predict(dataframe_without_class_and_time,dataframe_to_post_results,threshold,betaValues):
    for i in range(len(dataframe_to_post_results)):
        if logistic(dataframe_without_class_and_time.iloc[[i]].to_numpy().ravel() @ betaValues )>=threshold:
            dataframe_to_post_results.at[dataframe_to_post_results.index[i],"Predicted"]=1
        else: 
            dataframe_to_post_results.at[dataframe_to_post_results.index[i],"Predicted"]=0

def metrics(dataframe_to_post_the_results):
    #Various Metrics
    True_Positive=0
    True_Negative=0
    False_Positive=0
    False_Negative=0

    for row_index in range(len(dataframe_to_post_the_results)):
        if dataframe_to_post_the_results.loc[row_index, "Class"]==1 and dataframe_to_post_the_results.loc[row_index, "Predicted"]==1:
            True_Positive+=1
        elif dataframe_to_post_the_results.loc[row_index, "Class"]==0 and dataframe_to_post_the_results.loc[row_index, "Predicted"]==0:
            True_Negative+=1
        elif dataframe_to_post_the_results.loc[row_index, "Class"]==0 and dataframe_to_post_the_results.loc[row_index, "Predicted"]==1:
            False_Positive+=1  
        elif dataframe_to_post_the_results.loc[row_index, "Class"]==1 and dataframe_to_post_the_results.loc[row_index, "Predicted"]==0:
            False_Negative+=1

    Accuracy = (True_Positive+True_Negative)/(True_Positive+True_Negative+False_Positive+False_Negative)
    Error_Rate = 1-Accuracy
    Precision = True_Positive/(True_Positive+False_Positive)
    Sensitivity = True_Positive/(True_Positive+False_Negative)
    Negative_Predictive_Value = True_Negative/(False_Negative+True_Negative)
    Specificity = True_Negative/(False_Positive+True_Negative)
    F_1_Score = 2*True_Positive/(2*True_Positive+False_Positive+False_Negative)

    print("Size of data set is ", len(dataframe_to_post_the_results))
    print("True Positive amount is ", True_Positive)
    print("True Negative amount is ", True_Negative)
    print("False Positive amount is ", False_Positive)
    print("False Negative amount is " ,False_Negative)
    print("Accuracy is ",Accuracy)
    print("Error Rate is ", Error_Rate)
    print("Precision is ", Precision)
    print("Sensitivity is ", Sensitivity)
    print("Negative Predictive Value is ", Negative_Predictive_Value)
    print("Specificity is ", Specificity)
    print("F1 Score is ", F_1_Score)

def metrics2(dataframe_to_post_them_results):
    correct = (dataframe_to_post_them_results["Class"] == dataframe_to_post_them_results["Predicted"]).sum()
    total = len(df)
    print("How many incorrectly predicted:", total - correct)
    print("Portion correctly predicted:", correct / total)


def ALL_COLUMNS(DATAFRAME_INPUT):
    for i, col in enumerate(DATAFRAME_INPUT.columns, start=1):
        print(f"DATAFRAME_INPUT column {i}: {col}")



#HERE IS WHERE THE MAIN THINGS START. 
#I UNDERSAMPLED BY TAKING ALL THE 492 ROWS WITH CLASS==1 AND RANDOMLY TOOK 492  OTHER ROWS WITH CLASS==0 AND CREATED A NEW DATAFRAME
#FROM THIS NEWLY CREATED DATAFRAME, I GOT NEW BETA VALUES USING ITERATIVE REWEIGHTED LEAST SQUARES
#THEN I USED THE WHOLE ORIGINAL DATAFRAME AS A TEST SET
#RECEIVER OPERATOR CURVE'S AREA UNDER THE CURVE HAD A SCORE OF 0.98999

rng = np.random.default_rng()
seed = int(rng.integers(0, 1_000_000))
Balanced_data_frame = UndersampleDataFrame(df,492,seed)

###TRAIN
#DATAFRAME WITHOUT CLASS AND TIME
X9 = Balanced_data_frame.drop(columns=["Class", "Time"])
X9.insert(0, "constant", 1)
X9 = X9.reset_index(drop=True)

###TRAIN
#DATAFRAME TO ACTUALLY POST RESULTS ON PREDICTION WITH
X10 = Balanced_data_frame.drop(columns=["Time"])
X10["Predicted"]=0
X10 = X10.reset_index(drop=True)


#TEST
#DATAFRAME WITHOUT CLASS AND TIME
XTEST1 = df.drop(columns=["Class", "Time"])
XTEST1.insert(0, "constant", 1)
XTEST1 = XTEST1.reset_index(drop=True)

#DATAFRAME TO ACTUALLY POST RESULTS ON PREDICTION WITH
XTEST2 = df.drop(columns=["Time"])
XTEST2["Predicted"]=0
XTEST2 = XTEST2.reset_index(drop=True)

# print("THIS IS FOR THE TRAINING")
# beta_undersampling = IRLS(Balanced_data_frame)
# Predict(X9,X10,0.1,beta_undersampling)
# metrics(X10)
# metrics2(X10)

print("NOW FOR THE TEST")
Predict(XTEST1,XTEST2,0.1,beta_undersampling)
metrics(XTEST2)
metrics2(XTEST2)
#ALL_COLUMNS(XTEST2)

XTEST2_FEATURES = XTEST2.drop(columns=["Class", "Predicted"])
XTEST2_FEATURES.insert(0, "constant", 1)

XTEST2_TARGET = XTEST2[["Class"]]



z = XTEST2_FEATURES.to_numpy() @ beta_undersampling
probs = expit(z)
y_test = XTEST2["Class"].astype(int)
print("ROC AUC:", roc_auc_score(y_test, probs))
