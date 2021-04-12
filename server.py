from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, models
import keras
from keras.utils import to_categorical
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from PIL import Image
#from wand.image import Image
import base64
import glob

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

@app.route('/snakeFilewrite', methods = ['POST'])
def snake_write_to_file():
	print(request.form['stuff'])
	stuff = request.form['stuff']
	with open('snakeMoves.txt', 'a') as f:
		f.write(str(stuff)[1:-1] + '\n')
	return jsonify('bruh')

@app.route('/game', methods = ['GET'])
def game():
	return render_template('Game.html')
	
@app.route('/snake', methods = ['GET'])
def snake():
	return render_template('snake.html')

@app.route('/baseline', methods = ['GET'])
def baseline():
	baseline = keras.Sequential()
	baseline.add(keras.Input(shape=(3,)))
	baseline.add(layers.Dense(50, activation='relu'))
	baseline.add(layers.Dense(100, activation='softmax'))

	input = []
	output = []
	with open('G:\Desktop\game\moves.txt','r+') as file:
		for line in file.readlines():
			stripped = line.strip().split(',')
			stripped = [int(i) for i in stripped]
			output.append(stripped[0])
			input.append(stripped[1:4])
	#newAdam = optimizers.Adam(lr=.0000000000000000000000000001)
	baseline.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')
	baseline.fit(input, output, epochs=1000, steps_per_epoch=1000)
	baseline.save('baseline.h5')
	
	print(output)
	return jsonify(output)

@app.route('/baselinePredict', methods = ['POST'])
def baselinePredict():
	model = keras.models.load_model('baseline.h5')
	player_location = int(request.form['playerLocation'])
	cookie_location = int(request.form['cookieLocation'])
	hole_location = int(request.form['holeLocation'])
	print(int(request.form['playerLocation']))
	print(int(request.form['cookieLocation']))
	print(int(request.form['holeLocation']))
	prediction = model.predict([[player_location, cookie_location, hole_location]])
	print(np.argmax(prediction))
	return jsonify({'answer':str(np.argmax(prediction))})

@app.route('/snakeBase', methods = ['GET'])
def snakeBase():
	snakeBase = keras.Sequential()
	snakeBase.add(keras.Input(shape=(104,)))
	snakeBase.add(layers.Dense(51, activation='relu'))
	snakeBase.add(layers.Dense(102, activation='softmax'))

	input = []
	output = []
	with open('G:\Desktop\game\snakeMoves.txt','r+') as file:
		for line in file.readlines():
			stripped = line.strip().split(',')
			stripped = [int(i) for i in stripped]
			output.append(stripped[0])
			input.append(stripped[1:len(stripped)])
	snakeBase.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')
	snakeBase.fit(input, output, epochs=500, steps_per_epoch=300)
	snakeBase.save('snakeBase.h5')
	
	print(output)
	return jsonify(output)
	
@app.route('/snakeBasePredict', methods = ['POST'])
def snakeBasePredict():
	model = keras.models.load_model('snakeBase.h5')
	input = request.form['stuff']
	input = input[1:len(input)-1]
	input = input.split(",")
	for x in range(0,len(input)):
		input[x] = int(input[x])
	prediction = model.predict([input])
	print(np.argmax(prediction))
	return jsonify({'answer':str(np.argmax(prediction))})

@app.route('/readData', methods = ['POST'])
def readData():
	input = []
	output = []
	with open('moves.txt','r+') as file:
		for line in file.readlines():
			stripped = line.strip().split(',')
			stripped = [int(i) for i in stripped]
			output.append(stripped[0])
			input.append(stripped[1:4])
	return jsonify({'input':input, 'output':output})
	
@app.route('/convolutional', methods = ['GET'])
def convolutional():
	#cnn = models.Sequential()
	#cnn.add(layers.Conv2D(25, (3, 3), activation='relu', input_shape=(16, 16, 3)))
	#cnn.add(layers.MaxPooling2D((2, 2)))
	#cnn.add(layers.Flatten())
	#cnn.add(layers.Dense(32, activation='relu'))
	#cnn.add(layers.Dense(4))
	model = models.Sequential() 
  
	# 1st Convolutional Layer 
	model.add(layers.Conv2D(filters = 96, input_shape = (224, 224, 3),  kernel_size = (11, 11), strides = (4, 4),  padding = 'same')) 
	model.add(layers.Activation('relu')) 
	# Max-Pooling  
	model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid')) 
	# Batch Normalisation 
	model.add(layers.BatchNormalization()) 
	# 2nd Convolutional Layer 
	model.add(layers.Conv2D(filters = 256, kernel_size = (11, 11),  strides = (1, 1), padding = 'valid')) 
	model.add(layers.Activation('relu')) 
	# Max-Pooling 
	model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2),  padding = 'valid')) 
	# Batch Normalisation 
	model.add(layers.BatchNormalization()) 
	# 3rd Convolutional Layer 
	model.add(layers.Conv2D(filters = 384, kernel_size = (3, 3),  strides = (1, 1), padding = 'valid')) 
	model.add(layers.Activation('relu')) 
	# Batch Normalisation 
	model.add(layers.BatchNormalization()) 
	# 4th Convolutional Layer 
	model.add(layers.Conv2D(filters = 384, kernel_size = (3, 3),  strides = (1, 1), padding = 'valid')) 
	model.add(layers.Activation('relu')) 
	# Batch Normalisation 
	model.add(layers.BatchNormalization()) 
	# 5th Convolutional Layer 
	model.add(layers.Conv2D(filters = 256, kernel_size = (3, 3),  strides = (1, 1), padding = 'valid')) 
	model.add(layers.Activation('relu')) 
	# Max-Pooling 
	model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2),  padding = 'valid')) 
	# Batch Normalisation 
	model.add(layers.BatchNormalization()) 
	# Flattening 
	model.add(layers.Flatten()) 
	# 1st Dense Layer 
	model.add(layers.Dense(4096, input_shape = (224*224*3, ))) 
	model.add(layers.Activation('relu')) 
	# Add Dropout to prevent overfitting 
	model.add(layers.Dropout(0.4)) 
	# Batch Normalisation 
	model.add(layers.BatchNormalization())  
	# 2nd Dense Layer 
	model.add(layers.Dense(2500)) 
	model.add(layers.Activation('relu')) 
	# Add Dropout 
	model.add(layers.Dropout(0.4)) 
	# Batch Normalisation 
	model.add(layers.BatchNormalization()) 
	# Output Softmax Layer 
	model.add(layers.Dense(4)) 
	model.add(layers.Activation('softmax')) 
	model.compile(optimizer='adam', loss="categorical_crossentropy" , metrics=['accuracy'])
	input = []
	output = []
	datagen = ImageDataGenerator()
	train_it = datagen.flow_from_directory('images/', batch_size=24, target_size=(224, 224))
	model.fit(train_it, epochs=80, steps_per_epoch=30) 
	model.save('cnn.h5')
	return "deez nuts"
	
@app.route('/convolutionalPredict', methods = ['POST'])
def convolutionalPredict():
	model = keras.models.load_model('cnn.h5')
	photo = load_img("G:/Desktop/game/bruh/currentBoard.png", target_size=(224, 224))
	photo = img_to_array(photo)
	prediction = model.predict(np.array([photo]))
	print(np.argmax(prediction))
	folder = 'G:/Desktop/game/bruh'
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))
	return jsonify({'answer':str(np.argmax(prediction))})
	
	
@app.route('/snakeConvolutionalPredict', methods = ['POST'])
def snakeConvolutionalPredict():
	model = keras.models.load_model('cnn.h5')
	photo = load_img("G:/Desktop/game/bruh/currentSnakeBoard.png", target_size=(224, 224))
	photo = img_to_array(photo)
	prediction = model.predict(np.array([photo]))
	print(np.argmax(prediction))
	folder = 'G:/Desktop/game/bruh'
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))
	return jsonify({'answer':str(np.argmax(prediction))})
	
@app.route('/snakeCNN', methods = ['POST'])
def snakeCNN():
	model = keras.models.load_model('cnn.h5')
	photo = load_img("G:/Desktop/game/bruh/snakeBoard.png", target_size=(224, 224))
	photo = img_to_array(photo)
	prediction = model.predict(np.array([photo]))
	print(np.argmax(prediction))
	folder = 'G:/Desktop/game/bruh'
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))
	return jsonify({'answer':str(np.argmax(prediction))})
	
	