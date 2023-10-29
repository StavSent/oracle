import numpy as np
import tensorflow as tf
import math
from bson.objectid import ObjectId
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def RePAD(data, api_id, save=True):
	optimizer = Adam(learning_rate=0.15)
	callback = EarlyStopping(monitor="loss", patience=3)

	x = data[:len(data)-1]
	x = np.array([x]).transpose().astype(float)
	x = x.reshape(x.shape[0], x.shape[1], 1)

	y = data[1:len(data)]
	y = np.array([y]).transpose().astype(float)
	y = y.reshape(y.shape[0], y.shape[1], 1)

	model = Sequential()
	model.add(LSTM(units=10))
	model.add(Dense(1))
    
	model.compile(loss="mean_squared_error", optimizer=optimizer)
	model.fit(x, y, epochs=250, batch_size=1, callbacks=[callback], verbose=0)
	
	if save: model.save("./models/{}.keras".format(api_id))

	x_pred = data[len(data)-1]
	x_pred = np.array([x_pred]).transpose().astype(float)
	x_pred = x_pred.reshape(1, 1, 1)
	y_pred = model.predict(x_pred, verbose=0)

	return y_pred[0][0], model

def RePAD_predict(x_pred, api_id):
	model = tf.keras.models.load_model("./models/{}.keras".format(api_id))
	x_pred = x_pred.reshape(1, 1, 1)
	y_pred = model.predict(x_pred, verbose=0)
	return y_pred[0][0]

def AARE(u, u_hat):
	if (len(u) != len(u_hat)):
		return 0	# TODO: change to None
	
	return (1 / 3) * sum((abs(u[i] - u_hat[i])) / u[i] for i in range(len(u)))

def RePAD_threshold(db, api_id):
	responses = db.responses.find({ "_id": ObjectId(api_id) }, { "AARE": 1 }).limit(2880)
	aares = [res.get("AARE") for res in responses if res.get("AARE") is not None]

	# the following part is derived from RePAD2's implementation
	uAARE = (1 / len(aares)) * sum(aares)
	std = math.sqrt((1 / len(aares)) * sum((aare - uAARE)**2 for aare in aares))
	
	return uAARE+(3*std) 