import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras import backend as K

def big_sig(x):
	return 2 * tf.sigmoid(x)

def exp(x):
	return K.exp(x)