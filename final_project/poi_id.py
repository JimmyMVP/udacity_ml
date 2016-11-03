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

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import robust_scale
from sklearn.svm import SVC


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

print(data.shape)
## Scaling the data
data = np.array(robust_scale(data[:0.8*len(data), :]))

outlier_remover = SVC()

print(data.shape)
outlier_remover.fit(data[:,1:], data[:,0])
errors = np.absolute(outlier_remover.decision_function(data[:,1:])).reshape(data.shape[0],1)
print(errors.shape)
data_with_outlyingness = np.hstack([data, errors])

print(errors)

print(data_with_outlyingness.shape)

print("Before outlier removing: " + str(data.shape))
data = np.array(sorted(data_with_outlyingness, key=lambda x: x[0], reverse=True))[0:0.9*len(data),0:-1]
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

print("Decomposing with PCA...")

pca = PCA(n_components=int(len(features[0])*0.8))
pca.fit(features)

#features = pca.transform(features)

print("Training classifier...")
clf = DecisionTreeClassifier()


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

clf.fit(features_train, labels_train)

from sklearn.metrics import precision_score, recall_score, f1_score


predictions = clf.predict(features_test)

print("F1: " + str(f1_score(labels_test, predictions)))
print("Recall: " + str(recall_score(labels_test, predictions)))
print("Precision: " + str(precision_score(labels_test, predictions)))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)