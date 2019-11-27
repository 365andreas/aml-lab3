# remove noise
# baseline adjustment(?)
# feature extraction
# normalize features
# classification

# various lengths
# imbalanced dataset

from __future__ import division
from biosppy import storage
from biosppy.signals import ecg
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import classification_report, f1_score, confusion_matrix, balanced_accuracy_score
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import column_or_1d
from statistics import mean

def sort(val):
    return val[0]

def plot_signal(i):
	signal_one = series.iloc[i]
	signal_one = signal_one.dropna()
	print(signal_one)
	signal_one.plot()
	pyplot.savefig('class_1_' + str(i) + '.png')

# noise removal(?)
def extract_features_rpeaks(rpeaks, signal):
	#find mean Rampl
	r_ampl = signal[rpeaks]
	r_mean = np.mean(r_ampl)
	r_std = np.std(r_ampl)
	min_r = min(r_ampl)
	max_r = max(r_ampl)

	#find rr intervals
	rr_int = np.diff(rpeaks)
	rr_int_mean = np.mean(rr_int)
	rr_int_std = np.std(rr_int)
	rr_sqrd = (np.square(rr_int))
	rr_sum = np.sum(rr_sqrd)/rr_int.shape[0]
	rr_rms = math.sqrt(rr_sum)
	return [r_mean, r_std, min_r, max_r, rr_int_mean, rr_int_std, rr_rms]
	
def create_dataset(input_file):	
	# load raw ECG signal
	series = pd.read_csv(input_file, header=0, index_col=0, parse_dates=True, squeeze=True)
	n_samples = series.shape[0]
	stds=[]
	d_list = []
	column_names = ['r_ampl_mean', 'r_ampl_std', 'r_ampl_min', 'r_ampl_max', 'rr_mean', 'rr_std', 'rr_rms']
		
	for i in range(n_samples):
		print(i)
		signal = series.iloc[i]
		signal = signal.dropna()
		signal_array = np.asarray(signal)
		out = ecg.ecg(signal=signal_array, show=False, sampling_rate=300)
	
		heartbeats = out.as_dict()['templates']
		heartrate = out.as_dict()['heart_rate']
		rpeaks = out.as_dict()['rpeaks']
		features_for_s = extract_features_rpeaks(rpeaks, signal_array)
		d_list.append(features_for_s)
	df = pd.DataFrame(d_list, columns=column_names)
	return df	

x_train_init = create_dataset("X_train.csv")
x_train_init.to_csv("x_train_rpeaks.csv")
# load dataset
#x_train_init = pd.read_csv("x_train_rpeaks.csv")
y_train_init = pd.read_csv("y_train.csv")

#x_train_init = x_train_init.drop(x_train_init.columns[[0]], axis=1) 
y_train_init = y_train_init.drop('id', axis=1)
#x_train_init = x_train_init.drop('id', axis=1)

print(x_train_init)
# classifier
cv_results=[]
for c, gamma in [(c, g)
			for c in [1.0, 10.0, 100.0, 1000.0]
			for g in [0.00001, 0.0001, 0.001, 0.01, 0.1]]:

	print("--------------------- C = ", c, " gamma = ", gamma)	
	clf = SVC(C=c, kernel='rbf', gamma=gamma, class_weight='balanced', random_state=17)

	val_scores=[]
	N = 10
	kf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
	for train_index, test_index in kf.split(x_train_init, y_train_init):

		X_train_i, X_val_i = x_train_init.iloc[train_index], x_train_init.iloc[test_index]
		Y_train_i, Y_val_i = y_train_init.iloc[train_index], y_train_init.iloc[test_index]

		#Y_train_i = column_or_1d(Y_train_i, warn=True)

		model = clf
		model.fit(X_train_i, Y_train_i)

		Y_pred_val = model.predict(X_val_i)

		test_s = f1_score(Y_val_i, Y_pred_val, average='micro')
		val_scores.append(test_s)

	print(val_scores)
	print("mean: ", mean(val_scores))
	print("std: ", np.std(val_scores))
	cv_results.append([mean(val_scores), np.std(val_scores), c, gamma])

cv_results.sort(key=sort, reverse=True)
print(cv_results)

######################################################
'''
#best_result = cv_results[0]
#print(best_results)

model = SVC(C=10.0, kernel='rbf', gamma=1e-05, class_weight='balanced')
model.fit(x_train_init, y_train_init)
y_pred = model.predict(x_train_init)

print("Confusion Matrix of Training:")
print(confusion_matrix(y_train_init, y_pred, labels=[0, 1, 2, 3]))

print("Classification Report of Training:")
print(classification_report(y_train_init, y_pred, labels=[0, 1, 2, 3]))

print("F1:")
print(f1_score(y_train_init, y_pred, average='micro'))


# 5. Make predictions
x_test = create_dataset("X_test.csv")
print(x_test)
#x_test.to_csv("x_test_rpeaks.csv")
#x_test = pd.read_csv("x_test_rpeaks.csv")
#x_test = x_test.drop(x_test.columns[[0]], axis=1) 

y_test =  model.predict(x_test)
print(y_test)
Id=[]
for i in range(x_test.shape[0]):
	Id.append(i) 
df = pd.DataFrame(Id, columns=['id'])
df.insert(1, "y", y_test)
df.to_csv('solution_naive.csv', index=False)
'''
