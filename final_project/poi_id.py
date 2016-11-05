#!/usr/bin/python

import sys
import pickle
import sys
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments', 
                        'loan_advances', 'bonus', 'restricted_stock_deferred', 
                        'deferred_income', 'total_stock_value', 'expenses', 
                        'exercised_stock_options', 'other', 'long_term_incentive', 
                        'restricted_stock', 'director_fees'] 
email_features =['to_messages', 'email_address', 'from_poi_to_this_person', 
                'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

features_list = ['poi','salary', 'total_stock_value', 
        'director_fees', 'loan_advances', 'shared_receipt_with_poi', 'long_term_incentive', 'exercised_stock_options'] # You will need to use more features

features_list = ['poi'] + financial_features 
features_list = features_list + ['from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import robust_scale
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from numpy import linalg
import pdb

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

## Scaling the data
data = np.array(robust_scale(data))
scaler = MinMaxScaler()
scaler.fit(data)

print("Data shape: " + str(data.shape))
    
###Remove with KMeans

outlier_remover = KMeans(n_clusters=1)
outlier_remover.fit(data)

cluster_center = np.array(outlier_remover.cluster_centers_[0])
errors = np.sqrt(((data-cluster_center) ** 2).sum(1)).reshape(-1,1)

###Remove with linear regression



print(errors.shape)
data_with_outlyingness = np.hstack([data, errors])
data_with_errors = np.array(sorted(data_with_outlyingness, key=lambda x: x[-1], reverse=False))
print(data_with_outlyingness.shape)



print("Before outlier removing: " + str(data.shape))
#Sort ascending, the data points with the least error first
data = data_with_errors[0:1*len(data), 0:-1]
                       
print("After outlier removing: " + str(data.shape))


labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

print("Decomposing with PCA...")

pca = PCA(n_components=int(len(features[0])))

print("Training classifier...")
dt = RandomForestClassifier(warm_start=True)

from sklearn.model_selection import GridSearchCV

params = {
          "min_samples_split" : [5,10,12,17,20,22]
          }
grid = GridSearchCV(dt, params, cv=17, n_jobs=-1, refit=True, scoring='recall')

clf = Pipeline([('scaler', scaler), ('pca', pca),('grid', grid)])


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features, labels)

from sklearn.metrics import precision_score, recall_score, f1_score



#Plotting
pca_features = pca.transform(features)
predictions = clf.predict(features)

import matplotlib.pyplot as plt

colors = ['r', 'b']
for i, (x,y,col)  in enumerate(np.hstack([pca_features[:,0:2], predictions.reshape(-1,1)]).tolist()):
    plt.scatter(x,y, color = colors[predictions[i] == labels[i]])
#plt.show()

clf = Pipeline([('scale', scaler), ('pca', pca),('estimator', grid.best_estimator_)])

predictions = predictions.reshape(-1,1)
#print("F1: " + str(f1_score(labels, predictions)))
print("Recall: " + str(grid.best_score_))
#print("Precision: " + str(precision_score(labels, predictions)))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)