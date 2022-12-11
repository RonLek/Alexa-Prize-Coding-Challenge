# Load libraries
import flask
import pandas as pd
import tensorflow as tf
import tensorflow_text as text
from keras.models import load_model
import tensorflow_hub as hub
import numpy as np

# instantiate flask 
app = flask.Flask(__name__)

# load the model, and pass in the custom metric function
model = load_model('model_ic.h5', custom_objects={'KerasLayer': hub.KerasLayer})

# list of all labels
labels = ['answer_question_external_fact', 'answer_question_recipe_steps',
       'ask_question_ingredients_tools', 'ask_question_recipe_steps',
       'ask_student_question', 'chitchat', 'misc', 'request_next_step',
       'return_list_ingredients_tools', 'return_next_step', 'stop']

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {}

    params = flask.request.json

    predictions = tf.nn.softmax(model.predict([params['text']]))
    index = np.argmax(predictions)

    data["intent"] = labels[index]
    data["score"] = str(np.amax(predictions))

    # return a response in json format 
    return flask.jsonify(data)    

@app.route('/status', methods=['GET'])
def get_status():
    return flask.jsonify({
        'status' : 'alive'
    })

# start the flask app, allow remote connections 
app.run(host='0.0.0.0')

