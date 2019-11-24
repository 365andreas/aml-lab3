# remove noise
# baseline adjustment(?)
# feature extraction
# normalize features
# classification

# various lengths
# imbalanced dataset

from biosppy import storage
from biosppy.signals import ecg
import pandas as pd
from matplotlib import pyplot
import numpy as np

def plot_signal(i):
	signal_one = series.iloc[i]
	signal_one = signal_one.dropna()
	print(signal_one)
	signal_one.plot()
	pyplot.savefig('class_1_' + str(i) + '.png')

y_train = pd.read_csv("y_train.csv")
y_train.drop('id', axis=1)
y_train_list = list(y_train.y)

# load raw ECG signal
series = pd.read_csv('X_train.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
n_samples = series.shape[0]
stds=[]
d_list = []
for i in range(n_samples):
	mean_l = []
	signal = series.iloc[i]
	signal = signal.dropna()
	signal_array = np.asarray(signal)
	out = ecg.ecg(signal=signal, show=False, sampling_rate=300)
	out_t = out.as_dict()['templates']	
	pd.DataFrame(data=out_t)
	out_pd = pd.DataFrame(data=out_t)

	mean_l = [y_train_list[i]]
	means = list(out_pd.mean())
	for i in means:
		mean_l.append(i)
	#std_l = list(out_pd.std())
	#stds.append(std_l)
	d_list.append(mean_l)

print(d_list)
df = pd.DataFrame(d_list)
df.to_csv('mean_values.csv')







