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

def plot_signal(i):
	signal_one = series.iloc[i]
	signal_one = signal_one.dropna()
	print(signal_one)
	signal_one.plot()
	pyplot.savefig('class_1_' + str(i) + '.png')


# load raw ECG signal
series = pd.read_csv('X_train.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
signal = series.iloc[32]
signal = signal.dropna()
# process it and plot
out = ecg.ecg(signal=signal, show=False)
print(out)







