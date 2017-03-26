import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from keras.models import Sequential

def getLoss(metrics=None):
	'''
	given a Keras model's performance metrics, return the loss 
	'''
	return metrics[0]

def calcLoss(predicted, actual):
	'''
	manually calculate Poisson loss, given predicted and actual (ground-truth)
	values  
	'''
	return np.mean(np.mean(predicted - actual*np.log(predicted + 1e-07), axis=-1))

def getPreds(model, input, reshape=False):
	'''
	return the Keras model predictions on a given input
	-> reshape keyword is for Recurrent or Convolutional architectures
	'''
	if reshape:
		input = np.reshape(input, (input.shape[0], input.shape[1], 1))
	preds = model.predict(input)
	return np.reshape(preds, (preds.shape[0],))

def getNegLogLi(predicted, actual):
	'''
	given predicted and ground truth values, return the Negative Log Likelihood
	'''
	return np.sum((predicted - actual*np.log(predicted + 1e-07)))

def getR2(predicted, actual):
	'''
	return the r^2 value, given predicted and ground truth values 
	'''
	return (np.corrcoef(predicted, actual)[0,1]) ** 2

def visualize(preds, subset, actual, length=100):
	'''
	plot predictions from a subset of models against the true activity;
	-> default setting is to print from all models
	-> currently compares over first 100 time bins of test data


	'''
	for name, pred in preds.iteritems():
		if name in subset:
			plt.plot(pred[:length], label=name)
	plt.plot(actual[:length], label='Actual')
	plt.legend()
	plt.show()

def vis_metrics(hist):
	plt.plot(hist.history['loss'], label='loss')
	plt.plot(hist.history['val_loss'], label='validation loss')
	plt.legend()
	plt.title('Training and Validation Loss History')
	plt.xlabel('Epoch')
	plt.show()

def bar_comparison(data, title):
	'''
	compare model performances given the result data and title 
	(via bar graph)
	'''
	N = len(data)
	ind = np.arange(N)
	width = 0.5
	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, data.values(), width, color='b')

	ax.set_ylabel(title)
	ax.set_xticks(ind + 0.5 * width)
	ax.set_xticklabels(data.keys())

	plt.show()