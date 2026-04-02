# CC-Fraud-Project
Binary classification project for detecting fraudulent credit card transactions using anonymized, highly imbalanced tabular data. Implements baseline and tree-based machine learning models, class-imbalance handling, and precision–recall evaluation to assess fraud risk.

# Credit Card Fraud Detection

## Overview
This project uses a binary classification pipeline to detect fraudulent credit card transactions using anonymized, highly imbalanced tabular data. The goal is to use baseline and tree-based machine learning models, class-imbalance handling, and precision-recall evaluation to determine if a given transaction has a high likelihood of fraud. 

## Problem Statement
Credit card fraud detection is a persistent challenge for Canadian financial institutions. Fraudulent transactions are a tiny fraction of overall activity, making it difficult to improve detection without increasing false positives, frustrating and disrupting legitimate transactions for consumers. Yet fraud is disproportionately costly, resulting in hundreds of millions of dollars in annual losses, along with second-order operational and reputational costs for institutions that fail to detect it. This project addresses the inherent tradeoffs of fraud detection caused by asymmetric costs and high class-imbalance by applying baseline and tree-based machine learning to anonymized, highly imbalanced data. 

## Dataset
The dataset lists hundreds of thousands of anonymized credit card transactions (rows) described by time, transaction amount, numerical variables V1-V28, and the output variable Class (columns).  In the interest of anonymity, customer names are not included, and variables V1–V28 are renamed and transformed versions of the original transaction attributes. The Class variable is a binary that indicates whether a transaction is fraudulent (1) or non-fraudulent (0). Over 99% of transactions fall into the latter category, so the dataset shows high class imbalance. 

## Methodology
This project builds a binary classification model to detect credit card fraud. Logistic regression is poorly suited for this task due to the high class imbalance in this data (e.g. a model detecting 0% of credit card fraud would be nearly 100% accurate) and its inability to capture nonlinear relationships between the input variables and the output (Class) variable. Instead, logistic regression is used only as a measuring stick for performance. The primary focus is on tree-based machine learning models, which are trained with class weighting to correct for imbalanced data and can handle more complex relationships. During evaluation, a probability threshold for fraud is selected and adjusted to manage the trade-off between sensitivity to fraud and false positives. We later automate this process with a model (how? elaborate)

## Evaluation Metrics
This project’s evaluation metrics are precision, recall, and the F1-score measured against the baseline logistic regression model. Precision measures the share of flagged transactions that are true positives and recall measures the share of fraud that is detected by the model. The F1-score is a composite precision-recall measure that reflects the tradeoff between the two metrics. These metrics are used instead of accuracy because the latter fails under high class imbalance (since only a tiny share of transactions are fraudulent, a model can be highly accurate while missing all or most fraud) while the former provide a more comprehensive assessment of performance.

## Results
Summary of model performance and key findings.

## Limitations
Known constraints and shortcomings of the approach.

## Future Work
Potential improvements or extensions.

## Repository Structure
Explanation of folders and files.

## How to Run
Basic instructions to reproduce results.
