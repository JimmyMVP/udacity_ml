#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)





### it's all yours from here forward!  



from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

dt = DecisionTreeClassifier()

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, random_state=42, test_size=0.3)


Y_test = np.array(Y_test)

print(Y_test.size)
dt.fit(X_train, Y_train)



print("Decision tree accuracy: " + str(accuracy_score(dt.predict(X_test), Y_test)))
print("Confusion matrix:")
print(confusion_matrix(Y_test, dt.predict(X_test)))
print("Recall: " + str(recall_score(Y_test, dt.predict(X_test))))
print("Precision: " + str(precision_score(Y_test, dt.predict(X_test))))


