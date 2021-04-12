from flask import Flask, request, jsonify, render_template
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import keras
from keras.utils import to_categorical
import numpy as np
app = Flask(__name__)

@app.route('/filewrite', methods = ['POST'])
def write_to_file():
	print(request.form['bestMoveIndex'])
	best_move_index = request.form['bestMoveIndex']
	player_location = request.form['playerLocation']
	cookie_location = request.form['cookieLocation']
	hole_location = request.form['holeLocation']
	with open('moves.txt', 'a') as f:
		f.write(str(best_move_index) + ',' + str(player_location) + ',' + str(cookie_location) + ',' + str(hole_location) + '\n')
	return jsonify('bruh')

@app.route('/game', methods = ['GET'])
def game():
	return render_template('Game.html')

@app.route('/baseline', methods = ['GET'])
def baseline():
	baseline = keras.Sequential()
	baseline.add(keras.Input(shape=(3,)))
	baseline.add(layers.Dense(2, activation='relu'))
	baseline.add(layers.Dense(4, activation='softmax'))

	input = []
	output = []
	with open('G:\Desktop\game\moves2.txt','r+') as file:
		for line in file.readlines():
			stripped = line.strip().split(',')
			stripped = [float(i) for i in stripped]
			output.append(stripped[0] - 1)
			input.append(stripped[1:4])
	print("helllo")
	print(input)
	#newAdam = optimizers.Adam(lr=.0000000000000000000000000001)
	final_input=np.asarray(input)
	final_output=to_categorical(output,4)
	baseline.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')
	baseline.fit(final_input, final_output, epochs=100, steps_per_epoch=50)
	baseline.save('baseline.h5')
	
	print(output)
	return jsonify(output)

@app.route('/predict', methods = ['POST'])
def predict():
	model = keras.models.load_model('baseline.h5')
	player_location = int(request.form['playerLocation'])
	cookie_location = int(request.form['cookieLocation'])
	hole_location = int(request.form['holeLocation'])
	prediction = model.predict([[player_location, cookie_location, hole_location]])
	print(prediction)
	return jsonify(prediction)
