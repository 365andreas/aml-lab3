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
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix, balanced_accuracy_score
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import column_or_1d
from statistics import mean

import scipy.signal as signal
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#sample frequency
Fs = 300
T = 1.0 / Fs

def plot(raw, filtered):
	N = raw.size
	x = np.linspace(0.0, N*T, N)

	fig_td = plt.figure(figsize=(20,10))
	fig_td.canvas.set_window_title('Time domain signals')
	ax1 = fig_td.add_subplot(211)
	ax1.set_title('Before filtering')
	ax2 = fig_td.add_subplot(212)
	ax2.set_title('After filtering')
	ax1.plot(x,raw, color='r', linewidth=0.7)
	ax2.plot(x,filtered, color='g', linewidth=0.7);
	fig_td.savefig('filtering_class0.png')


# noise reduction: https://github.com/tejasa97/ECG-Signal-Processing
def multiple_filters(y):

	#band_filt = np.array([45, 55])
	#b, a = signal.butter(2, band_filt/(Fs/2), 'bandstop', analog=False)
	b, a = signal.butter(4, 50/(Fs/2), 'low')

	###ax3.plot(w, 20 * np.log10(abs(h)))
	#Compute filtered signal
	tempf = signal.filtfilt(b,a, y)
		                                                       ### Compute Kaiser window co-effs to eliminate baseline drift noise ###
	nyq_rate = Fs/ 2.0
	# The desired width of the transition from pass to stop.
	width = 5.0/nyq_rate
	# The desired attenuation in the stop band, in dB.
	ripple_db = 60.0
	# Compute the order and Kaiser parameter for the FIR filter.
	O, beta = signal.kaiserord(ripple_db, width)
	# The cutoff frequency of the filter.
	cutoff_hz = 4.0
		                                                        ###Use firwin with a Kaiser window to create a lowpass FIR filter.###
	taps = signal.firwin(O, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)
	# Use lfilter to filter x with the FIR filter.
	y_filt = signal.lfilter(taps, 1.0, tempf)
	return y_filt

def lp_filter_demo(raw):
	N  = 4    # Filter order - CHANGE
	Wn = 0.1 # Cutoff frequency - CHANGE
	B, A = signal.butter(N, Wn, output='ba', btype='low') # use another filter
	smooth_data = signal.filtfilt(B,A, raw)
	return smooth_data

def sort(val):
    return val[0]

def plot_signal(i):
	signal_one = series.iloc[i]
	signal_one = signal_one.dropna()
	print(signal_one)
	signal_one.plot()
	pyplot.savefig('class_1_' + str(i) + '.png')


def extract_features(rpeaks, heartbeat, signal):
	#QRS(?)

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

	#heartbeats
	heartbeat = pd.DataFrame(heartbeat)
	heartbeat = heartbeat.mean()
	heartbeat = heartbeat.to_numpy()
	hb_sq = np.square(heartbeat)
	wv = np.sum(hb_sq)

	return [r_mean, r_std, min_r, max_r, rr_int_mean, rr_int_std, rr_rms, wv]

def create_dataset(input_file):	
	# load raw ECG signal
	series = pd.read_csv(input_file, header=0, index_col=0, parse_dates=True, squeeze=True)
	n_samples = series.shape[0]
	stds=[]
	d_list = []
	column_names = ['r_ampl_mean', 'r_ampl_std', 'r_ampl_min', 'r_ampl_max', 'rr_mean', 'rr_std', 'rr_rms', 'wavelet_energy']
		
	for i in range(n_samples):
		print(i)
		signal = series.iloc[i]
		signal = signal.dropna()
		signal_array = multiple_filters(signal) # denoising
		#signal_array = np.asarray(signal)
		out = ecg.ecg(signal=signal_array, show=False, sampling_rate=300)
	
		heartbeats = out.as_dict()['templates']
		heartrate = out.as_dict()['heart_rate']
		rpeaks = out.as_dict()['rpeaks']
		features_for_s = extract_features(rpeaks, heartbeats, signal_array)
		d_list.append(features_for_s)
	df = pd.DataFrame(d_list, columns=column_names)
	return df	

#x_train_init = create_dataset("X_train.csv")
#x_train_init.to_csv("x_train_denoised.csv")
# load dataset


x_train_init = pd.read_csv("x_train_manual_features.csv")
x_train_init = x_train_init.drop(x_train_init.columns[[0]], axis=1) 

print(x_train_init.isnull())
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
x_train_filled = imputer.fit_transform(x_train_init)
x_train_init = pd.DataFrame(x_train_filled)
print(x_train_init.isnull())

column_names = ['RRint_mean', 'RRint_sd', 'RRint_10', 'RRint_90', \
           'Rampl_mean', 'Rampl_sd', 'Rampl_10', 'Rampl_90', \
           'Qampl_mean', 'Qampl_sd', 'Qampl_10', 'Qampl_90', \
           'Sampl_mean', 'Sampl_sd', 'Sampl_10', 'Sampl_90', \
           'QRSts_mean', 'QRSts_sd', 'QRSts_10', 'QRSts_90', \
           'HR_mean', 'HR_sd', 'HR_10', 'HR_90']

#for i in column_names:
#	index = x_train_init[i].index[x_train_init[i].apply(np.isnan)]
#	print(index)

scaler = StandardScaler()
x_train_new = scaler.fit_transform(x_train_init)
cols = list(x_train_init.columns.values)
x_train_init = pd.DataFrame(data=x_train_new, columns=cols)


y_train_init = pd.read_csv("y_train.csv")
y_train_init = y_train_init.drop('id', axis=1)
'''
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
'''

######################################################

#best_result = cv_results[0]
#print(best_results)

model = SVC(C=1000.0, kernel='rbf', gamma=0.01, class_weight='balanced')
model.fit(x_train_init, y_train_init)
y_pred = model.predict(x_train_init)

print("Confusion Matrix of Training:")
print(confusion_matrix(y_train_init, y_pred, labels=[0, 1, 2, 3]))

print("Classification Report of Training:")
print(classification_report(y_train_init, y_pred, labels=[0, 1, 2, 3]))

print("F1:")
print(f1_score(y_train_init, y_pred, average='micro'))


# 5. Make predictions
#x_test = create_dataset("x_test_manual_features.csv")
#x_test.to_csv("x_test_rpeaks.csv")

x_test = pd.read_csv("x_test_manual_features.csv")
x_test = x_test.drop(x_test.columns[[0]], axis=1) 

x_test_filled = imputer.transform(x_test)
x_test = pd.DataFrame(x_test_filled)

x_test = scaler.transform(x_test)
print(x_test)

y_test =  model.predict(x_test)
print(y_test)
Id=[]
for i in range(x_test.shape[0]):
	Id.append(i) 
df = pd.DataFrame(Id, columns=['id'])
df.insert(1, "y", y_test)
df.to_csv('solution_svm.csv', index=False)
