import numpy as np
import tensorflow as tf
import math
from bson.objectid import ObjectId
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def RePAD_lstm(data, api_id, save=True):
	optimizer = Adam(learning_rate=0.05)
	callback = EarlyStopping(monitor="loss", patience=10)

	# data = [(d - max_value) / (max_value - min_value) for d in data]
	# data = [(d / 1000) for d in data]
	norm = np.linalg.norm(data[:len(data)-1])
	data = np.array(data) / norm

	x = data[:len(data)-1]
	x = np.array(x).astype(float).reshape(1, len(x), 1)

	y = data[-1]
	y = np.array(y).astype(float).reshape(1, 1, 1)

	model = Sequential()
	model.add(LSTM(units=10))
	model.add(Dense(1))
    
	model.compile(loss="mape", optimizer=optimizer)
	f = model.fit(x, y, epochs=50, callbacks=[callback], verbose=0)	
	y_pred = model(x).numpy()[0][0]
	# y_pred = ((y_pred_norm) * (max_value - min_value)) + max_value

	print("y_predicted from lstm: {}".format(y_pred))

	if save: model.save("./models/{}.keras".format(api_id))

	return float(y_pred * norm), norm, model

def RePAD_predict(x_pred, norm, api_id):
	model = tf.keras.models.load_model("./models/{}.keras".format(api_id))
	x_pred = np.array(x_pred) / norm
	# x_pred = [(x - max_value) / (max_value - min_value) for x in x_pred]
	# x_pred = [x / 1000 for x in x_pred]

	x_pred = np.array(x_pred).astype(float).reshape(1, len(x_pred), 1)
	y_pred = model(x_pred).numpy()[0][0]
	# y_pred = ((y_pred_norm) * (max_value - min_value)) + max_value

	return float(y_pred * norm)

def AARE(u, u_hat):
	print(u)
	print(u_hat)
	if (len(u) != len(u_hat)):
		return 0	# TODO: change to None
	
	return (1 / 3) * sum(((abs(u[i] - u_hat[i])) / u[i]) for i in range(len(u)))

def RePAD_threshold(db, api_id):
	responses = db.responses.find({ "api": ObjectId(api_id) }, { "AARE": 1 }).sort("createdAt", -1)
	# responses = db.responses.find({ "api": ObjectId(api_id) }, { "AARE": 1 }).sort("createdAt", -1).limit(2880)
	aares = [res.get("AARE") for res in responses if (res.get("AARE") is not None)]

	# the following part is derived from RePAD2's implementation
	uAARE = (1 / len(aares)) * sum(aares)
	std = math.sqrt((1 / len(aares)) * sum((aare - uAARE)**2 for aare in aares))

	print("μ: {}".format(uAARE))
	print("std: {}".format(std))
	print("thd: {}".format(uAARE+(3*std)))
	
	return uAARE+(3*std) 