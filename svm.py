import preprocess
import sys
import getopt
import os
import math
import operator
import random
from sklearn import svm
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from colorama import init
from termcolor import colored
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import numpy as np
import pickle
from sklearn.svm import SVC
import pandas as pd

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# with open("train_tweets.txt", "r") as f:
#     test_data = f.readlines()
#     file_data[i] = preprocess.processAll(test_data)

def refine_all_tweets(file_data):

	for i, tweet in zip(range(len(file_data)),file_data):
		file_data[i] = preprocess.processAll(tweet)


#y_names is name of the categories
def test_classifier(X, y, clf, test_size=0.4, y_names=None, confusion=False):
	#train-test split
	print ('test size is: %2.0f%%' % (test_size*100))
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size,random_state=42)
	clf.fit(X_train, y_train)
	y_predicted = clf.predict(X_test)
	print (y_predicted)
	acc = sklearn.metrics.accuracy_score(y_test,y_predicted)

	print ("Accuracy is ",acc)
	if not confusion:
		print (colored('Classification report:', 'magenta', attrs=['bold']))
		print (sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names))
	else:
		print (colored('Confusion Matrix:', 'magenta', attrs=['bold']))
		print (sklearn.metrics.confusion_matrix(y_test, y_predicted))


	#title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' % classifier.best_estimator_.gamma
	#estimator = SVC(kernel='linear', gamma=classifier.best_estimator_.gamma)
	#plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
	#plt.show()

def split_dataset(files_data,files_target,test_size,i):
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(files_data, files_target, test_size=test_size, random_state=i)
	return X_train, X_test, y_train, y_test

def bag_of_words(files_data_train,files_data_test):
	"""
	Converts a list of strings (which are loaded from files) to a BOW representation of it
	parameter 'files_data' is a list of strings
	returns a `scipy.sparse.coo_matrix`
	"""
	count_vector = sklearn.feature_extraction.text.CountVectorizer()
	#print count_vector.fit(files_data)
	word_train =  count_vector.fit_transform(files_data_train)
	word_test = count_vector.transform(files_data_test)
	print (len(count_vector.get_feature_names()))
	return word_train,word_test

def feature_extract(is_idf,word_train,word_test):
	if is_idf:
		tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True)
	else:
		tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=False)
	X_tr = tf_transformer.fit(word_train).transform(word_train)
	X_te = tf_transformer.transform(word_test)
	return X_tr,X_te

def main():
	# directory='Tweet_Data'
	# files = sklearn.datasets.load_files(directory)
	test_size=0.1

	# num_folds = 1
	# for i in range(num_folds):
	# 	

	# with open("train_tweets.txt", "r") as f:
	# 	test_data = f.readlines()
	# 	file_data[i] = preprocess.processAll(test_data)
	X_train = pd.read_csv("train_tweets.txt", sep="\t", header=None)
	train_tweets = X_train[1]
	train_label = X_train[0]
	# X_train = "train_tweets.txt"

	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(train_tweets, train_label, test_size=0.33, random_state=42)
	
	# X_train,X_test,y_train,y_test = split_dataset(files.data,files.target, test_size,i)
	# X_test = "test_tweets_unlabeled.txt"
	refine_all_tweets(X_train)
	refine_all_tweets(X_test)

	word_train,word_test = bag_of_words(X_train,X_test)
	is_idf = False
	X_tr,X_te = feature_extract(is_idf,word_train, word_test)


		#create classifier

		#clf = sklearn.naive_bayes.MultinomialNB()
		#clf = tree.DecisionTreeClassifier()

		# Classifier - Algorithm - SVM
	# 	# fit the training dataset on the classifier
	# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
	# SVM.fit(X_train,y_train)

	clf = sklearn.svm.LinearSVC()
	clf.fit(X_tr,y_train)
	# save the model to disk
	filename = 'finalized_model.sav'
	pickle.dump(model, open(filename, 'wb'))
	y_pred = clf.predict(X_te)
	y_names=files.target_names
	acc = sklearn.metrics.accuracy_score(y_test, y_pred)
	print (sklearn.metrics.classification_report(y_test, y_pred, target_names=y_names))
	print ("accuracy=",acc)



if __name__ == "__main__":
    main()