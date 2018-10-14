import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from PIL import Image

data = np.genfromtxt('/Users/zhujunli/Desktop/features/feature_MaxPool_2.csv', delimiter=',')

label = np.genfromtxt('/Users/zhujunli/Desktop/features/label.csv', delimiter=',')

data = data / data.max()
data = data.reshape(32, 32*32*128)

label = label[~np.isnan(label)]
print data.shape
print label.shape

kf = KFold(n_splits=10,random_state=None, shuffle=True)
kf.get_n_splits(data)
print(kf)

clf_tree = tree.DecisionTreeClassifier(criterion='entropy') 
accuracy = 0
for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data = data[train_index], data[test_index]
    train_label, test_label = label[train_index], label[test_index]
    clf_tree = clf_tree.fit(train_data, train_label)
    
    score = clf_tree.score(test_data, test_label)  
   
    accuracy += score
print "tree_criterion: entropy ", accuracy/10 #62.5%

clf_NB = model = GaussianNB()
accuracy = 0

for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data = data[train_index], data[test_index]
    train_label, test_label = label[train_index], label[test_index]
    clf_NB = clf_NB.fit(train_data, train_label)
    
    score = clf_NB.score(test_data, test_label)  
    
    accuracy += score
print "Naive Bayes: default ", accuracy/10  #50%


clf_RF = RandomForestClassifier()
accuracy = 0


for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data = data[train_index], data[test_index]
    train_label, test_label = label[train_index], label[test_index]
    clf_RF = clf_RF.fit(train_data, train_label)
    
    score = clf_RF.score(test_data, test_label)  
    
    accuracy += score
print "Random Forest: default ", accuracy/10 #80%

clf_GB= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
accuracy = 0


for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data = data[train_index], data[test_index]
    train_label, test_label = label[train_index], label[test_index]
    clf_GB = clf_GB.fit(train_data, train_label)
    
    score = clf_GB.score(test_data, test_label)  
    
    accuracy += score
print "GradientBoosting: n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0:", accuracy/10 #60%



clf = svm.LinearSVC()
accuracy = 0
for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data = data[train_index], data[test_index]
    train_label, test_label = label[train_index], label[test_index]
    
    clf = clf.fit(train_data, train_label)
    
    score = clf.score(test_data, test_label)  
    accuracy += score
print "linear svm: L2 ", accuracy/10  #97%

