# -*- coding: utf-8 -*-


from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
import sklearn
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from mlxtend.plotting import plot_confusion_matrix 
from mlxtend.classifier import StackingClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sn
import imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict


# get the dataset
df = pd.read_csv('C:/Users/Ian/Desktop/Twitter Ids/all_model_data.csv', index_col=False)
df.set_index('user', inplace = True)
#check for inf values due to division by 0
df.columns.to_series()[np.isinf(df).any()]
print(df.head())

# replace infinity values with 0's
df.replace([np.inf, -np.inf], np.nan,inplace=True)
df.replace(np.nan, 0,inplace=True)

#drop botscore and predecessor columns
df2 = df.drop(['botscore', 'pred_link_count'], axis = 1)
#feature data
X = df2.loc[:, df2.columns != 'bot_indicator)']
#target data
y = df2['bot_indicator)']

#spitting the data into testing and training data. parameters are for 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=0,stratify=y)


#initialise classifiers
LR = LogisticRegression()
KNC = KNeighborsClassifier()   
RF = RandomForestClassifier() 
GB = GradientBoostingClassifier()


## Logisitc regression

model_lr = LR.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)
lr_prob = model_lr.predict_proba(X_test)

#preparing ROC values

from sklearn.metrics import roc_curve

# roc curve for models
fprlr, tplr, threshlr = roc_curve(y_test,lr_prob[:,1], pos_label=1)


# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr_lr, p_tpr_lr, _lr = roc_curve(y_test, random_probs, pos_label=1)

#calculatting AUC score

from sklearn.metrics import roc_auc_score

# auc scores
auc_scorelr = roc_auc_score(y_test, lr_prob[:,1])


print(auc_scorelr)

Plotting ROC curve

# matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.plot(fprlr, tplr, linestyle='--',color='orange', label='Logistic Regression')

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();

#Cross validation



#skf = StratifiedKFold(n_splits=4)
y_pred = cross_val_predict(LR, X, y, cv = 10)

print(classification_report(y, y_pred))


print(metrics.confusion_matrix(y_test, pred_lr))
print(sn.heatmap(metrics.confusion_matrix(y_test, pred_lr),annot=True))

from sklearn.metrics import recall_score
tpr = recall_score(y_test, pred_lr)   # it is better to name it y_test 
# to calculate, tnr we need to set the positive label to the other class
# I assume your negative class consists of 0, if it is -1, change 0 below to that value
tnr = recall_score(y_test, pred_lr, pos_label = 0) 
fpr = 1 - tnr
fnr = 1 - tpr

print(fpr)

import imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


#feature data
#X_list = df2.loc[:, df2.columns != 'bot_indicator)'].values
#target data
#y_list = df2['bot_indicator)'].values

#X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X, y,test_size=0.3,random_state=0,stratify=y)

sm = SMOTE(sampling_strategy=0.5,random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 


print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 

#
model_lr_sm = LR.fit(X_train_res, y_train_res.ravel())
pred_lr_sm = model_lr_sm.predict(X_test)
#lr_prob_sm = model_lr.predict_proba(X_test)

#y_pred = cross_val_predict(LR, X, y, cv = 10)

print(classification_report(y_test, pred_lr_sm))


# transform the dataset
#oversample = SMOTE()
#Xs, ys = oversample.fit_resample(X_list, y_list)

#model = LR
#over = SMOTE(sampling_strategy=0.8)
#under = RandomUnderSampler(sampling_strategy=0.5)
#steps = [('over', over), ('under', under), ('model', model)]
#pipeline = Pipeline(steps=steps)
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores = cross_val_score(pipeline, Xs, ys, scoring='roc_auc', cv=cv, n_jobs=-1)
#print('Mean ROC AUC: %.3f' % mean(scores))


## K nearest neighbours

#Training KNeighborsClassifier
model_knc = KNC.fit(X_train, y_train)   # fitting Training Set 
pred_knc = model_knc.predict(X_test) 
knc_prob = model_knc.predict_proba(X_test)

preparing ROC values


# roc curve for models
fprknc, tpknc, threshknc = roc_curve(y_test,knc_prob[:,1], pos_label=1)


# roc curve for tpr = fpr 
random_probs_knc = [0 for i in range(len(y_test))]
p_fpr_knc, p_tpr_knc, _knc = roc_curve(y_test, random_probs_knc, pos_label=1)

calculatting AUC score

# auc scores
auc_scoreknc = roc_auc_score(y_test, knc_prob[:,1])


print(auc_scoreknc)

Plotting ROC curve


plt.style.use('seaborn')

# plot roc curves
plt.plot(fprknc, tpknc, linestyle='solid',color='blue', label='Kneares Neighbours')

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();

#cross validation

#skf = StratifiedKFold(n_splits=4)
y_pred = cross_val_predict(KNC, X, y, cv = 10)

print(classification_report(y, y_pred))


print(metrics.confusion_matrix(y_test, pred_knc))
print(sn.heatmap(metrics.confusion_matrix(y_test, pred_knc),annot=True))

from sklearn.metrics import recall_score
tprknc = recall_score(y_test, pred_knc)   # it is better to name it y_test 
# to calculate, tnr we need to set the positive label to the other class
# I assume your negative class consists of 0, if it is -1, change 0 below to that value
tnrknc = recall_score(y_test, pred_knc, pos_label = 0) 
fprknc = 1 - tnrknc
fnrknc = 1 - tprknc

print(fprknc )

sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 


print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 

#
model_knc_sm = KNC.fit(X_train_res, y_train_res.ravel())
pred_knc_sm = model_knc_sm.predict(X_test)
#lr_prob_sm = model_lr.predict_proba(X_test)

#y_pred = cross_val_predict(LR, X, y, cv = 10)

print(classification_report(y_test, pred_knc_sm))


## Random Forrest

model_rf = RF.fit(X_train, y_train)   # fitting Training Set 
pred_rf = model_rf.predict(X_test) 
rf_prob = model_rf.predict_proba(X_test)

Preparing ROC Curve

# roc curve for models
fprrf, tprf, threshrf = roc_curve(y_test,rf_prob[:,1], pos_label=1)


# roc curve for tpr = fpr 
random_probs_rf = [0 for i in range(len(y_test))]
p_fpr_rf, p_tpr_rf, _rf = roc_curve(y_test, random_probs_rf, pos_label=1)

calculating AUC

# auc scores
auc_scorerf = roc_auc_score(y_test, rf_prob[:,1])


print(auc_scorerf)

plotting ROC curve

plt.style.use('seaborn')

# plot roc curves
plt.plot(fprrf, tprf, linestyle='--',color='red', label='Random Forret')

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();

#skf = StratifiedKFold(n_splits=4)
y_pred = cross_val_predict(RF, X, y, cv = 10)

print(classification_report(y, y_pred))


print(metrics.confusion_matrix(y_test, pred_rf))
print(sn.heatmap(metrics.confusion_matrix(y_test, pred_rf),annot=True))

## Gradient booster

model_gb = GB.fit(X_train, y_train)
pred_gb = model_gb.predict(X_test)
gb_prob = model_gb.predict_proba(X_test)

preparing ROC curve

# roc curve for models
fprgb, tpgb, threshgb = roc_curve(y_test,gb_prob[:,1], pos_label=1)


# roc curve for tpr = fpr 
random_probs_gb = [0 for i in range(len(y_test))]
p_fpr_gb, p_tpr_gb, _gb = roc_curve(y_test, random_probs_gb, pos_label=1)

calculating AUC score

# auc scores
auc_scorekgb = roc_auc_score(y_test, gb_prob[:,1])


print(auc_scorekgb)

Plotting ROC curve

plt.style.use('seaborn')

# plot roc curves
plt.plot(fprgb, tpgb, linestyle='--',color='red', label='Gradint booster')

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();

Cros validation



y_pred = cross_val_predict(GB, X, y, cv = 10)

print(classification_report(y, y_pred))

print(metrics.confusion_matrix(y_test, pred_rf))
print(sn.heatmap(metrics.confusion_matrix(y_test, pred_gb),annot=True))

Comparsion of ROC curves

plt.style.use('seaborn')

# plot roc curves
plt.plot(fprgb, tpgb, linestyle='--',color='yellow', label='Gradint booster')
plt.plot(fprrf, tprf, linestyle='--',color='green', label='Random Forret')
plt.plot(fprknc, tpknc, linestyle='solid',color='blue', label='Knearest Neighbours')
plt.plot(fprlr, tplr, linestyle='--',color='orange', label='Logistic Regression')

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();

## Stacking classifier

Preparing stacked classifier

from mlxtend.classifier import StackingCVClassifier


sclf = StackingCVClassifier(classifiers = [LR,KNC, RF, GB],
                            shuffle = False,
                            use_probas = True,
                            cv = 10,
                            meta_classifier = LR)

#Create list to store classifiers

classifiers = {"LR": LR,
               "KNC": KNC,
               "RF": RF,
               "GB": GB,
               "Stack": sclf}

#Training stacked classifiers

for key in classifiers:
    # Get classifier
    classifier = classifiers[key]
    
    # Fit classifier
    classifier.fit(X_train, y_train)
        
    # Save fitted classifier
    classifiers[key] = classifier

#Get result of the stacking

# Get results
results = pd.DataFrame()
for key in classifiers:
    # Make prediction on test set
    y_pred = classifiers[key].predict_proba(X_test)[:,1]
    
    # Save results in pandas dataframe object
    results[f"{key}"] = y_pred

# Add the test set to the results object
results["Target"] = y_test

results.head(5)

#Comparison of classifiers

# Probability Distributions Figure
# Set graph style
import seaborn as sns
from sklearn import metrics
sns.set(font_scale = 1)
sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4'})

# Plot
f, ax = plt.subplots(figsize=(13, 4), nrows=1, ncols = 5)

for key, counter in zip(classifiers, range(5)):
    # Get predictions
    y_pred = results[key]
    
    # Get AUC
    auc = metrics.roc_auc_score(y_test, y_pred)
    textstr = f"AUC: {auc:.3f}"

    # Plot false distribution
    false_pred = results[results["Target"] == 0]
    sns.distplot(false_pred[key], hist=True, kde=False, 
                 bins=int(25), color = 'red',
                 hist_kws={'edgecolor':'black'}, ax = ax[counter])
    
    # Plot true distribution
    true_pred = results[results["Target"] == 1]
    sns.distplot(results[key], hist=True, kde=False, 
                 bins=int(25), color = 'green',
                 hist_kws={'edgecolor':'black'}, ax = ax[counter])
    
    
    # These are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # Place a text box in upper left in axes coords
    ax[counter].text(0.05, 0.95, textstr, transform=ax[counter].transAxes, fontsize=14,
                    verticalalignment = "top", bbox=props)
    
    # Set axis limits and labels
    ax[counter].set_title(f"{key} Distribution")
    ax[counter].set_xlim(0,1)
    ax[counter].set_xlabel("Probability")

# Tight layout
plt.tight_layout()

# Save Figure
plt.savefig("Probability Distribution for each Classifier.png", dpi = 1080)

model_sclf = sclf.fit(X_train, y_train)
pred_sclf = model_sclf.predict(X_test)
sclf_prob = model_sclf.predict_proba(X_test)

prepatring ROC cureve

# roc curve for models
fprstc, tpstc, threshstc = roc_curve(y_test,sclf_prob[:,1], pos_label=1)


# roc curve for tpr = fpr 
random_probs_stc = [0 for i in range(len(y_test))]
p_fpr_stc, p_tpr_stc, _stc = roc_curve(y_test, random_probs_stc, pos_label=1)

# auc scores
auc_scorestc = roc_auc_score(y_test, sclf_prob[:,1])


print(auc_scorestc)

plt.style.use('seaborn')

# plot roc curves
plt.plot(fprstc, tpstc, linestyle='--',color='red', label='Stacked')

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();

plt.style.use('seaborn')

# plot roc curves
plt.plot(fprgb, tpgb, linestyle='--',color='orange', label='Gradint booster')
plt.plot(fprrf, tprf, linestyle='--',color='green', label='Random Forret')
plt.plot(fprknc, tpknc, linestyle='--',color='blue', label='Knearest Neighbours')
plt.plot(fprlr, tplr, linestyle='--',color='purple', label='Logistic Regression')
plt.plot(fprstc, tpstc, linestyle='--',color='red', label='Stacked')

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


y_pred = cross_val_predict(sclf, X, y, cv = 10)

print(classification_report(y, y_pred))

print(metrics.confusion_matrix(y_test,  pred_sclf))
print(sn.heatmap(metrics.confusion_matrix(y_test, pred_sclf),annot=True))

calculating FPR

from sklearn.metrics import recall_score
tprsclf = recall_score(y_test, pred_sclf)   # it is better to name it y_test 
# to calculate, tnr we need to set the positive label to the other class
# I assume your negative class consists of 0, if it is -1, change 0 below to that value
tnrsclf = recall_score(y_test, pred_sclf, pos_label = 0) 
fprsclf = 1 - tnrsclf
fnrsclf = 1 - tprsclf

print(fprsclf)

sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 


print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 

#
model_sclf_sm = sclf.fit(X_train_res, y_train_res.ravel())
pred_sclf_sm = model_sclf_sm.predict(X_test)
#lr_prob_sm = model_lr.predict_proba(X_test)

#y_pred = cross_val_predict(LR, X, y, cv = 10)

print(classification_report(y_test, pred_sclf_sm))


model_sclf = sclf.fit(X_train, y_train)
pred_sclf = model_sclf.predict(X_test)
sclf_prob = model_sclf.predict_proba(X_test)



## RF meta classsifier


y_pred = cross_val_predict(sclf, X, y, cv = 10)

print(classification_report(y, y_pred))

sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 


print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 

#
model_sclf_sm = sclf.fit(X_train_res, y_train_res.ravel())
pred_sclf_sm = model_sclf_sm.predict(X_test)
#lr_prob_sm = model_lr.predict_proba(X_test)

#y_pred = cross_val_predict(LR, X, y, cv = 10)

print(classification_report(y_test, pred_sclf_sm))


sclf2 = StackingCVClassifier(classifiers = [KNC, RF, GB],
                            shuffle = False,
                            use_probas = True,
                            cv = 5,
                            meta_classifier = RF)

classifiers2 = {"KNC": KNC,
               "RF": RF,
               "GB": GB,
               "Stack": sclf2}

for key in classifiers2:
    # Get classifier
    classifier2 = classifiers2[key]
    
    # Fit classifier
    classifier2.fit(X_train, y_train)
        
    # Save fitted classifier
    classifiers2[key] = classifier2

model_sclf2 = sclf2.fit(X_train, y_train)
pred_sclf2 = model_sclf2.predict(X_test)
sclf_prob2 = model_sclf2.predict_proba(X_test)

y_pred2 = cross_val_predict(sclf2, X, y, cv = 10)

print(classification_report(y, y_pred2))




