#%% md
# # <u>***ANOMALY DETECTION : CREDIT CARD FRAUD***</u>
# 
# **Problem: Anomaly Detection in Credit Card Transactions**
#%% md
# # Table of Contents
# 
# **[1. SUMMARY](#1.-SUMMARY)**
# 
# **[2. DATA](#2.-DATA)**
# 
# > [2.1 About the Features](#2.1-About-the-Features)
# 
# > [2.2 Problem](#2.2-Problem )
# 
# > [2.3 Target Variable](#2.3-Target-Variable)
# 
# **[3. ANALYSIS](#3.-ANALYSIS)**
# 
# > [3.1 Reading the Data](#3.1-Reading-the-Data)
# 
# > [3.2 Exploratory Data Analysis (EDA) & Visualization](#3.2-Exploratory-Data-Analysis-(EDA)-&-Visualization)
# 
# >> [3.2.1 The Examination of Target Variable](#3.2.1-The-Examination-of-Target-Variable)
# 
# >> [3.2.2 The Examination of Numerical Features](#3.2.2-The-Examination-of-Numerical-Features)
# 
# **[4. FEATURE SCALING](#4.-FEATURE-SCALING)**
# 
# **[5. MODELLING & MODEL PERFORMANCE](#5.-MODELLING-&-MODEL-PERFORMANCE)**
# 
# > [5.1 The Implementation of Isolation Forest](#5.1-The-Implementation-of-Isolation-Forest)
# 
# > [5.2 The Implementation of One-Class SVM](#5.2-The-Implementation-of-One-Class-SVM)
# 
# > [5.3 The Implementation of Autoencoders](#5.3-The-Implementation-of-Autoencoders)
# 
# > [5.4 The Implementation of Local Outlier Factor (LOF)](#5.4-The-Implementation-of-Local-Outlier-Factor-(LOF))
# 
# > [5.5 The Implementation of DBSCAN](#5.5-The-Implementation-of-DBSCAN)
# 
# **[6. THE COMPARISON OF MODELS](#6.-THE-COMPARISON-OF-MODELS)**
# 
# **[7. CONCLUSION](#7.-CONCLUSION)**
# 
# **[8. REFERENCES](#8.-REFERENCES)**
# 
# > [Note](#Note)
#%% md
# # 1. SUMMARY 
# 
# I employed Exploratory Data Analysis (EDA) and various Anomaly Detection Algorithms, including Isolation Forest, One-Class SVM,  Autoencoders, Local Outlier Factor (LOF) and DBSCAN to examine the dataset 'Credit Card Fraud Detection' from the Kaggle website, which is labeled as 'creditcard.csv'. 
# 
# (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud )
# 
# I attempted to explore the dataset comprehensively, examining various aspects and visualizing as much as possible to gain insights into the data. I employed ---five(5)--  Anomaly Detection Algorithms.
#%%
# installing the necessary libraries

!pip install tensorflow
!pip install plotly
!pip install plotly cufflinks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
%matplotlib inline

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout  
from tensorflow.keras.regularizers import l2 

import cufflinks as cf
import plotly.offline as pyo
from plotly.offline import iplot
import plotly.graph_objs as go
pyo.init_notebook_mode(connected=True)
cf.go_offline()

palette = ['#00777F', '#5BABF5', '#AADEFE', '#EAAC9F', '#8AA0AF']
sns.set_theme(context='notebook', palette=palette, style='darkgrid')

# Suppress the FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
# Function for determining the number and percentages of missing values

def missing (df):
    num_of_missing = df.isnull().sum().sort_values(ascending=False)
    percentage_of_missing = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_values = pd.concat([num_of_missing, percentage_of_missing], axis=1, keys=['Num_of_missing', 'Percentage_of_missing'])
    return missing_values
#%%
# Function for insighting summary information about the column

def first_looking(col):
    print("column name    : ", col)
    print("--------------------------------")
    print("per_of_nulls   : ", "%", round(df[col].isnull().sum()/df.shape[0]*100, 2))
    print("num_of_nulls   : ", df[col].isnull().sum())
    print("num_of_uniques : ", df[col].nunique())
    print(df[col].value_counts(dropna = False))
#%% md
# # 2. DATA
# 
# ## 2.1 About the Features
# 
# The dataset contains transactions made by credit cards in September 2013 by European cardholders.
# 
# This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# * It contains only numerical input variables which are the result of a PCA transformation.
# * Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. 
# * Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset.
# * The feature 'Amount' is the transaction Amount, this feature can be used for example-dependent cost-sensitive learning. 
# * Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
#%% md
# ## 2.2 Problem
# 
# This is an anomaly detection problem where I will make predictions on the target variable "Class". Subsequently, I will compare the predictions of five machine learning algorithms and attempt to determine the best-performing model.
#%% md
# ## 2.3 Target Variable
# 
# In a fraud detection problem where the target variable is Class (indicating whether a transaction is fraudulent or not), the challenge is to identify fraud (anomalies) among a majority of normal transactions. When applying different anomaly detection algorithms such as Isolation Forest, One-Class SVM, Autoencoders, Local Outlier Factor (LOF), and DBSCAN, the target variable is not explicitly used during model training (since these are generally unsupervised learning methods). Instead, these algorithms learn from the structure of the data to detect outliers, which can later be compared with the target variable to evaluate performance.
# 
# Fraudulent transactions (the target variable "Class" with value 1) are identified as anomalies or outliers by each of these algorithms. The common theme across these methods is that fraud is treated as data that deviates significantly from the majority of transactions (which are normal).
# The target variable is found by comparing the algorithm's prediction of whether a transaction is normal or anomalous with the actual labels, allowing we to measure the model's effectiveness at detecting fraud.
#%% md
# # 3. ANALYSIS
#%% md
# ## 3.1 Reading the Data
#%%
df = pd.read_csv('creditcard.csv')
#%%
df.head()
#%% md
# ## 3.2 Exploratory Data Analysis (EDA) & Visualization
#%%
df
#%%
df.head()
#%%
df.tail(10)
#%%
df.sample(10)
#%%
df.columns
#%%
print("My dataset consists of ", df.shape[0], "rows and", df.shape[1], "columns")
#%%
df.info()
#%%
df.describe().T
#%%
df.nunique()
#%%
# to find how many unique values object features have

for col in df.select_dtypes(include=[np.number]).columns:
  print(f"{col} has {df[col].nunique()} unique value")
#%%
missing (df)
#%% md
# ### 3.2.1 The Examination of Target Variable
#%%
first_looking("Class")
#%%
print(df["Class"].value_counts())
df["Class"].value_counts().plot(kind="pie", autopct='%1.1f%%', figsize=(10,10));
#%%
# discovering the numbers and percentages of frauded and NOT frauded customers

y = df['Class']
print(f'Percentage of Frauded Customer: % {round(y.value_counts(normalize=True)[1]*100,2)} --> \
({y.value_counts()[1]} cases for Frauded Customer)\nPercentage of NOT Frauded Customer: % {round(y.value_counts(normalize=True)[0]*100,2)} --> ({y.value_counts()[0]} cases for NOT Frauded Customer)')
#%%
# discovering the statistical values over other variables of NOT frauded customers 

df[df['Class']==0].describe().T.style.background_gradient(subset=['mean','std','50%','count'], cmap='RdPu')
#%%
# discovering the statistical values over other variables of frauded customers 

df[df['Class']==1].describe().T.style.background_gradient(subset=['mean','std','50%','count'], cmap='RdPu')
#%%
# Spliting Dataset into numeric & categoric features

numerical= df.drop(['Class'], axis=1).select_dtypes('number').columns

categorical = df.select_dtypes('object').columns

print(f'Numerical Columns:  {df[numerical].columns}')
print('\n')
print(f'Categorical Columns: {df[categorical].columns}')
#%% md
# ### 3.2.2 The Examination of Numerical Features
#%%
# Define a function to plot histograms for each feature

def plot_feature_histograms(data, features, rows=10, cols=3, figsize=(15, 40)):
    # Set up the figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Loop through features and create histograms
    for i, feature in enumerate(features):
        sns.histplot(data[feature], ax=axes[i], kde=False, bins=30)
        axes[i].set_title(f'Histogram of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')

    # Remove any unused subplots if the number of features is less than available subplots
    if len(features) < len(axes):
        for j in range(len(features), len(axes)):
            fig.delaxes(axes[j])

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

# Feature names (excluding the last column as target)
features = numerical

# Call the function to plot the histograms
plot_feature_histograms(df, features, rows=10, cols=3, figsize=(15, 40))
#%% md
# [Table of Contents](#Table-of-Contents)
#%% md
# # 4. FEATURE SCALING
#%%
# Create a copy of the DataFrame to avoid changing the original

df_transformed = df.copy()

# Function to handle log transformation for skewed data
def log_transform_skewed(column):
    # For positive and zero values (log1p avoids log(0) errors)
    transformed = np.where(column >= 0, np.log1p(column), -np.log1p(-column))
    return transformed

# Compute skewness before transformation
skewness_before = df.skew()

# Apply transformation to skewed columns
for col in features:
    if abs(df[col].skew()) > 0.75:  # Threshold for skewness
        df_transformed[col] = log_transform_skewed(df[col])

# Compute skewness after transformation
skewness_after = df_transformed.skew()

# Compare skewness before and after
skewness_comparison = pd.DataFrame({
    'Skewness Before': skewness_before,
    'Skewness After': skewness_after
})

# Print the comparison
skewness_comparison
#%%
# Set up the figure; 10 rows (10*3=30 subplots), adjust as needed

fig, axes = plt.subplots(10, 3, figsize=(15, 40))  # Adjust rows to fit all features

# Flatten axes array to loop through easily
axes = axes.flatten()

# Plot each feature in a separate subplot
for i, feature in enumerate(features):
    sns.histplot(df_transformed[feature], ax=axes[i], kde=False, bins=30)
    axes[i].set_title(f'{feature} after Transformation')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

# Remove any unused subplots if features < 30
for i in range(len(features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
#%%
# Separate features and target

X = df_transformed[numerical]
y = df_transformed.Class

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#%%
X
#%%
y
#%%
X_scaled
#%% md
# [Table of Contents](#Table-of-Contents)
#%% md
# # 5. MODELLING & MODEL PERFORMANCE
#%% md
# ## 5.1 The Implementation of Isolation Forest
#%% md
# **Isolation Forest:**
# 
# **Perspective:** This algorithm isolates observations by randomly selecting features and then splitting the data. Anomalies (fraudulent transactions) are more easily separated (isolated) because they tend to be rare and have different feature values from normal transactions. Each transaction is assigned an anomaly score, and those with high scores are predicted as fraud.
# 
# **Target Insight:** Fraudulent transactions are typically the ones that require fewer splits to be isolated, thus receiving higher anomaly scores.
#%%
# Initialize the Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=101)   # high contamination to catch more fraud transactions

# Fit the model and predict (returns -1 for anomalies and 1 for normal data)
iso_preds = iso_forest.fit_predict(X_scaled)

# Convert -1 (anomalies) to 1 (fraud) and 1 (normal) to 0 (non-fraud)
iso_preds = [1 if x == -1 else 0 for x in iso_preds]

# Evaluate the results
print(classification_report(y, iso_preds))
roc_auc = roc_auc_score(y, iso_preds)
print("ROC AUC Score: ", roc_auc)
#%%
# Custom color palette
colors = ['#CFEEF0', '#00777F']
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

# Plot the confusion matrix
cm = confusion_matrix(y, iso_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap=custom_cmap, fmt='g')

# Add labels, title, and axis ticks
plt.title('Confusion Matrix ')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'])
plt.yticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'])
plt.show()
#%% md
# ## 5.2 The Implementation of One-Class SVM
#%% md
# **One-Class SVM:**
# 
# **Perspective:** One-Class SVM models the majority class (normal transactions) by creating a decision boundary around it in a high-dimensional space. Points that fall outside this boundary are considered anomalies (fraudulent).
# 
# **Target Insight:** Transactions far from the decision boundary are labeled as fraud, while those within the boundary are labeled as normal.
#%%
# Initialize One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.05)

# Fit the model and predict (returns -1 for anomalies and 1 for normal data)
svm_preds = oc_svm.fit_predict(X_scaled)

# Convert -1 (anomalies) to 1 (fraud) and 1 (normal) to 0 (non-fraud)
svm_preds = [1 if x == -1 else 0 for x in svm_preds]

# Evaluate the results
print(classification_report(y, svm_preds))
roc_auc = roc_auc_score(y, svm_preds)
print("ROC AUC Score: ", roc_auc)
print("Confusion Matrix:")
#%%
# Plot the confusion matrix
cm = confusion_matrix(y, svm_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap=custom_cmap, fmt='g')

# Add labels, title, and axis ticks
plt.title('Confusion Matrix ')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'])
plt.yticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'])
plt.show()
#%% md
# ## 5.3 The Implementation of Autoencoders
#%% md
# **Autoencoders:**
# 
# **Perspective:** Autoencoders learn to compress (encode) and then reconstruct (decode) data. They minimize the reconstruction error for normal data. Fraudulent transactions, being different from the majority, have higher reconstruction errors.
# 
# **Target Insight:** Transactions with high reconstruction errors are flagged as anomalies (fraud), as the model struggles to accurately rebuild data that deviates from the normal pattern.
#%%
# Define the autoencoder model
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(encoded)
    encoded = Dense(8, activation='relu', kernel_regularizer=l2(0.001))(encoded)
    
    # Latent space
    latent = Dense(4, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(8, activation='relu', kernel_regularizer=l2(0.001))(latent)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(decoded)
    decoded = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    return autoencoder

# Build and compile the model
autoencoder = build_autoencoder(X_scaled.shape[1])
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

# Train the model on normal transactions (non-fraudulent class, y == 0)
X_train = X_scaled[y == 0]
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.1)

# Calculate reconstruction error for all transactions
reconstructed = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

# Set a threshold for anomaly detection
threshold = np.percentile(mse, 90)  # Adjust threshold (90th percentile)
autoen_preds = np.where(mse > threshold, 1, 0)  # 1: anomaly (fraud), 0: normal

# Evaluate the model
print(classification_report(y, autoen_preds))
roc_auc = roc_auc_score(y, autoen_preds)
print("ROC AUC Score: ", roc_auc)
#%%
# Plot the confusion matrix
cm = confusion_matrix(y, autoen_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap=custom_cmap, fmt='g')

# Add labels, title, and axis ticks
plt.title('Confusion Matrix ')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'])
plt.yticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'])
plt.show()
#%% md
# ## 5.4 The Implementation of Local Outlier Factor (LOF)
#%% md
# **Local Outlier Factor (LOF):**
# 
# **Perspective:** LOF measures the local density of data points compared to their neighbors. Anomalies (fraudulent transactions) are expected to have a much lower density than normal transactions.
# 
# **Target Insight:** Transactions that are located in sparse regions (with few close neighbors or much farther from their neighbors) are flagged as fraud.
#%%
# Initialize Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=10, contamination=0.05)

# Predict (returns -1 for anomalies and 1 for normal data)
lof_preds = lof.fit_predict(X_scaled)

# Convert -1 (anomalies) to 1 (fraud) and 1 (normal) to 0 (non-fraud)
lof_preds = [1 if x == -1 else 0 for x in lof_preds]

# Evaluate the results
print(classification_report(y, lof_preds))
roc_auc = roc_auc_score(y, lof_preds)
print("ROC AUC Score: ", roc_auc)
print("Confusion Matrix:")
print(confusion_matrix(y, lof_preds))
#%%
# Plot the confusion matrix
cm = confusion_matrix(y, lof_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap=custom_cmap, fmt='g')

# Add labels, title, and axis ticks
plt.title('Confusion Matrix ')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'])
plt.yticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'])
plt.show()
#%% md
# ## 5.5 The Implementation of DBSCAN
#%% md
# **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
# 
# **Perspective:** DBSCAN clusters data based on density. Transactions that do not belong to any cluster and are considered noise points are likely to be anomalies (fraud).
# 
# **Target Insight:** Fraudulent transactions appear as noise or outliers, meaning they don’t fit well into any cluster of normal transactions.
#%%
# Initialize DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Fit and predict (labels different clusters, outliers labeled as -1)
dbscan_preds = dbscan.fit_predict(X_scaled)

# Convert -1 (anomalies) to 1 (fraud) and others to 0 (non-fraud)
dbscan_preds = [1 if x == -1 else 0 for x in dbscan_preds]

# Evaluate the results
print(classification_report(y, dbscan_preds))
roc_auc = roc_auc_score(y, dbscan_preds)
print("ROC AUC Score: ", roc_auc)
print("Confusion Matrix:")
print(confusion_matrix(y, dbscan_preds))
#%%
# Plot the confusion matrix
cm = confusion_matrix(y, dbscan_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap=custom_cmap, fmt='g')

# Add labels, title, and axis ticks
plt.title('Confusion Matrix ')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'])
plt.yticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'])
plt.show()
#%% md
# # 6. THE COMPARISON OF MODELS
#%% md
# | Algorithms_Results  | Precision  | Recall    | F1-score   | ROC AUC Score   |
# |---------------------|------------|-----------|------------|-----------------|
# |Isolation Forest	  |  3.00%	   |  85.00%   |  6.00%	    | 90.151%         |
# |---------------------|------------|-----------|------------|-----------------|
# |One-Class SVM	      |  3.00%	   |  86.00%   |  6.00%	    | 90.354%         |
# |---------------------|------------|-----------|------------|-----------------|
# |Autoencoders	      |  1.00%	   |  87.00%   |  3.00%	    | 88.359%         |
# |---------------------|------------|-----------|------------|-----------------|
# |Local Outlier(LOF)	  |  0.00%	   |  11.00%   |  1.00%	    | 53.196%         |
# |---------------------|------------|-----------|------------|-----------------|
# |DBSCAN	              |  0.00%	   |  97.00%   |  0.00%	    | 56.839%         |   |
#%% md
# **1. Isolation Forest**
# 
# *Precision:* 3.00% (very low) — Among the transactions detected as fraud, only 3% are actually fraudulent, suggesting many false positives.
# 
# *Recall:* 85.00% (high) — The model successfully identifies 85% of the fraudulent transactions, meaning it’s good at catching fraud but at the cost of precision.
# 
# *F1-Score:* 6.00% (low) — The low F1 score shows the imbalance between Precision and Recall.
# 
# *ROC AUC Score:* 90.151% — This high score indicates good discrimination between fraud and non-fraud, even though Precision is low.
# 
# **2. One-Class SVM**
# 
# *Precision:* 3.00% (very low) — Like Isolation Forest, Precision is low, suggesting many false positives.
# 
# *Recall:* 86.00% (high) — Slightly higher than Isolation Forest, it captures a significant portion of fraudulent transactions.
# 
# *F1-Score:* 6.00% (low) — Similar to Isolation Forest, indicating a low balance between Precision and Recall.
# 
# *ROC AUC Score:* 90.354% — This score is slightly higher than Isolation Forest, indicating that it can separate fraud and non-fraud relatively well.
# 
# **3. Autoencoders**
# 
# *Precision:* 1.00% (very low) — Very few of the predicted fraud cases are actual fraud.
# 
# *Recall:* 87.00% (high) — High recall shows that Autoencoders identify most of the fraudulent transactions.
# 
# *F1-Score:* 3.00% (very low) — The low F1 score again indicates poor balance.
# 
# *ROC AUC Score:* 88.359% — The score is lower than Isolation Forest and One-Class SVM but still reasonable.
# 
# **4. Local Outlier Factor (LOF)**
# 
# *Precision:* 0.00% — The model fails to achieve any true positives, indicating it may not be effective for this dataset.
# 
# *Recall:* 11.00% — Very low recall means it misses most fraudulent transactions.
# 
# *F1-Score:* 1.00% — The lowest F1 score, showing poor performance overall.
# 
# *ROC AUC Score:* 53.196% — The low ROC AUC score confirms that LOF does not effectively distinguish between fraud and non-fraud transactions in this case.
# 
# **5. DBSCAN**
# 
# *Precision:* 0.00% — Like LOF, DBSCAN has no true positives, indicating no effective fraud detection.
# 
# *Recall:* 97.00% — Very high recall, suggesting it identifies nearly all fraud but likely with high false positives.
# 
# *F1-Score:* 0.00% — Due to the zero precision, the F1 score is 0.
# 
# *ROC AUC Score:* 56.839% — While better than LOF, the ROC AUC is still low, indicating poor performance for fraud detection.
# 
# **Summary**
# * *Isolation Forest* and *One-Class SVM* appear to be the best algorithms here, with high Recall and reasonable ROC AUC scores. However, both have low Precision, meaning they tend to mislabel normal transactions as fraud.
# * *Autoencoders* perform moderately well in terms of Recall but fall short in Precision.
# * *Local Outlier Factor* and *DBSCAN* perform poorly on this dataset, as evidenced by their low Precision, F1-scores, and ROC AUC values, indicating they are not suitable for this particular problem.
#%% md
# # 7. CONCLUSION
#%% md
# * For this fraud detection problem, **Isolation Forest** and **One-Class SVM** emerged as the best-performing algorithms among the five evaluated. Both methods demonstrate high Recall (85% and 86%, respectively) and ROC AUC scores (over 90%), indicating strong ability to detect fraud cases and reasonably distinguish between fraudulent and non-fraudulent transactions. However, their low Precision values (3%) reveal a significant number of false positives — normal transactions labeled as fraud — which could result in a high number of unnecessary alerts or investigations.
# 
# * **Autoencoders** also show high Recall (87%) but even lower Precision (1%), making them less reliable due to more frequent false positives. While Autoencoders demonstrate fair discrimination capability (ROC AUC of 88%), the imbalance between Precision and Recall reduces their practical value in this case.
# 
# * **Local Outlier Factor (LOF)** and **DBSCAN** underperform, with low Precision, Recall, F1 scores, and ROC AUC values. These algorithms struggle to effectively distinguish fraud in this dataset, as evidenced by their near-zero Precision and F1-scores. They may not be suitable for this problem, given their high error rates in identifying fraudulent transactions.
#%% md
# # Note
#%% md
# * Given the high Recall and ROC AUC of Isolation Forest and One-Class SVM, these algorithms are currently the most viable options for this fraud detection problem. However, to reduce false positives and improve Precision, further tuning or exploring hybrid approaches (such as combining anomaly detection with supervised learning for refinement) may be beneficial. Additionally, implementing a threshold or post-processing step to manage the trade-off between Precision and Recall could help make these models more practical for real-world use.
#%% md
# # 8. REFERENCES
# * https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# * https://www.kaggle.com/code/annastasy/anomaly-detection-credit-card-fraud/notebook
# * https://www.kaggle.com/code/kachalisarma20/credit-card-anomaly-detection#Model-Building-and-prediction
# * https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets#Anomaly-Detection:
# * https://www.kaggle.com/code/sugataghosh/anomaly-detection-in-credit-card-transactions#Implementing-Anomaly-Detection
# 
#%% md
# [Table of Contents](#Table-of-Contents)