from statistics import mean
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score
from flask import Flask, request, jsonify

app = Flask(__name__)

custom_objects = {'r2_score': r2_score}
model = tf.keras.models.load_model('ML_Program', custom_objects=custom_objects)

@app.route('/')
def home():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Access form data
        data = request.get_json()
        # Process the data
        #red = data['red']
        #ir = data['ir']

        #some sensor data is reversed
        ir = data['red']
        red = data['ir']

        prediksi = model.predict(np.array([mean(red), mean(ir)]))
        output = round(float(prediksi[0][0]), 2)
        x = {
            "output" : output
        }

        # Do something with the data
        return jsonify(x)
    
    else:
        return 'Invalid request method'
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)