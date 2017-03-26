import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as spio 
from numpy.linalg import inv, norm
from pyglmnet import GLM

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, SimpleRNN, LSTM, Convolution1D, Flatten
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.optimizers import RMSprop, SGD
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping, LearningRateScheduler
import time 

def getBaseline(train, test):
	'''
	Inputs: training set, testing set 
	Outputs: spike predictions for test set 
	note: every other model function returns the model
	'''
	avg_rate = float(np.sum(train)) / len(train)
	preds = np.zeros((len(test),))
	preds.fill(avg_rate)
	return preds

def buildGLM(data, iters=1000):
	'''
	returns GLM model set to run for iters iterations
	'''
	reg_lambda = np.logspace(np.log(1e-6), np.log(1e-6), 100, base=np.exp(1))
	glm_poissonexp = GLM(distr='poisson', verbose=False, alpha=.05, 
		max_iter=iters, learning_rate=2e-1, score_metric='pseudo_R2',
		reg_lambda=reg_lambda, eta=4.0)
	return glm_poissonexp

def getSTA(data):
	'''
	returns whitened sta for input data
	'''
	Xdsgn, sps, X_val, y_val, X_test, y_test = data
	nsp = int(np.sum(sps))
	nT = Xdsgn.shape[0]

	# spike triggered avg
	sta = np.dot(Xdsgn.T, sps) / nsp
	# whitened sta
	wsta = np.dot(inv(np.dot(Xdsgn.T, Xdsgn)), sta) * nsp
	return wsta

def getWts(model):
	'''
	given a model, returns a tuple of its Weights and Biases
	'''
	wts = []
	for layer in model.layers:
		wts.append(layer.get_weights())
	biases = []
	weights = []
	for i in xrange(len(wts)):
		weights.append(wts[i][0])
		biases.append(wts[i][1])
	weights = np.asarray(weights)
	weights = np.reshape(weights, (weights.shape[1],))
	biases = np.asarray(biases)[0]
	return weights, biases

def runGLM(model, data):
	'''
	fit GLM model and return GLM predictions on test data and loss 
	'''
	Xdsgn, sps, X_val, y_val, X_test, y_test = data
	# fitting
	model.fit(Xdsgn, sps)
	pGLMconst = model[-1].fit_['beta0']
	pGLMfilt = model[-1].fit_['beta']
	# predictions
	ratepred_pGLM = np.exp(pGLMconst + np.dot(Xdsgn,pGLMfilt))
	y_pred_GLM = np.exp(pGLMconst + np.dot(X_test, pGLMfilt))
	y_pred_GLM = np.reshape(y_pred_GLM, (y_pred_GLM.shape[0],))
	# loss 
	GLM_loss = np.mean(np.mean(y_pred_GLM - y_test * np.log(y_pred_GLM + 1e-07), axis=-1))
	return y_pred_GLM, GLM_loss

def buildFC(nlin='sigmoid', optim='SGD'):
	'''
	returns a 2-Layer Fully-Connected Feedfoward network model, built in Keras
	-> nlin = the nonlinearity, optim = the optimization algorithm 
	-> current architecture: 128 - 64 hidden units
	'''
	model = Sequential()
	model.add(Dense(output_dim=128, input_dim=20, activation='relu'))
	model.add(Dense(output_dim=64, input_dim=128, activation='relu'))
	model.add(Dense(output_dim=1, activation='softplus'))
	model.compile(loss='poisson', optimizer=optim, metrics=['mean_squared_error'])
	print(model.summary())
	return model 

def buildZero(nlin='sigmoid', optim='SGD'):
	'''
	returns a 0-layer network model, built in Keras
	-> nlin = the nonlinearity, optim = the optimization algorithm 
	'''
	model = Sequential()
	model.add(Dense(output_dim=1, input_dim=20, activation=nlin))
	model.compile(loss='poisson', optimizer=optim, metrics=['mean_squared_error'])
	print(model.summary())
	return model

def buildRNN(nlin='sigmoid', optim='SGD'):
	'''
	returns a 1-layer Vanilla Recurrent Network model, built in Keras
	-> nlin = the nonlinearity, optim = the optimization algorithm 
	-> current architecture: recurrent (64) - dense(64) 
	'''
	model = Sequential()
	# recurrent layer 
	model.add(SimpleRNN(output_dim=64, input_dim=1, input_length=20))
	model.add(Dense(output_dim=64, input_dim=64, activation=nlin))
	model.add(Dense(output_dim=64, input_dim=64, activation=nlin))
	model.add(Dense(1, activation=nlin))
	model.compile(loss='poisson', optimizer=optim, metrics=['mean_squared_error'])
	print(model.summary())
	return model

def buildLSTM(nlin='sigmoid', optim='SGD'):
	'''
	returns a 1-layer LSTM Recurrent Network model, built in Keras
	-> nlin = the nonlinearity, optim = the optimization algorithm 
	-> current architecture: 64 hidden units 
	'''
	model = Sequential()
	# recurrent layer 
	model.add(LSTM(64, input_dim=1, input_length=20))
	model.add(Dense(1, activation=nlin))
	sgd = SGD(lr = 0.02)
	model.compile(loss='poisson', optimizer=sgd, metrics=['mean_squared_error'])
	print(model.summary())
	return model

def buildConvNet(nlin='sigmoid', optim='SGD'):
	'''
	returns a 1-layer Convolutional Network model, built in Keras
	-> nlin = the nonlinearity, optim = the optimization algorithm 
	-> current architecture: Conv (128) - Max Pool - Dense (64)
	'''
	model = Sequential()
	model.add(Convolution1D(128, 1, border_mode='valid', input_dim=1, input_length=20))
	model.add(MaxPooling(3))
	model.add(Flatten())
	model.add(Dense(output_dim=64, activation=nlin))
	model.add(Dense(output_dim=1, activation=nlin))
	model.compile(loss='poisson', optimizer=optim, metrics=['mean_squared_error'])
	print(model.summary())
	return model

def reshape3D(data):
	'''
	Reshapes data to accomodate Recurrent and Convolutional network requirements
	'''
	X_train, y_train, X_val, y_val, X_test, y_test = data
	# input shape: (nb_samples, nb_timesteps, input_dim)
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
	X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
	# output: (nb_samples, output_dim)
	y_train = np.reshape(y_train, (len(y_train), 1))
	y_test = np.reshape(y_test, (len(y_test), 1))
	y_val = np.reshape(y_val, (len(y_val), 1))

	data3d = X_train, y_train, X_val, y_val, X_test, y_test
	return data3d

def step_decay(epoch):
	'''
	given a training epoch, returns a prescribed learning rate
	-> current setting: halve the learning rate every 7 epochs 
	'''
	initial_lrate = 0.01
	drop = 0.5
	epochs_drop = 7.0
	lrate = initial_lrate * drop ** np.floor((1+epoch)/epochs_drop)
	return float(lrate)

def runModel(model, data, epoch=15, reshape=False, drop=False):
	'''
	run a Keras model on the given data for the given number of epochs
	-> the reshape keyword is set to 'True' if using a Recurrent or 
	   Convolutional architecture
	-> returns a tuple containing the training history of the model and 
	   the final values for the performance metrics (i.e. loss and MSE)
	'''
	if reshape:
		X_train, y_train, X_val, y_val, X_test, y_test = reshape3D(data)
	else:
		X_train, y_train, X_val, y_val, X_test, y_test = data

	lrate = LearningRateScheduler(step_decay)
	val_data = X_val, y_val
	early_stopping = EarlyStopping(monitor='val_loss', patience=3) # stop early if loss goes up for 3 epochs
	if drop:
		cbacks = [lrate, early_stopping]
	else:
		cbacks = [early_stopping]

	hist = model.fit(X_train, y_train, nb_epoch=epoch, batch_size=32, verbose=2, validation_data=val_data, callbacks=cbacks)
	loss_and_metrics = model.evaluate(X_test, y_test)
	return hist, loss_and_metrics
