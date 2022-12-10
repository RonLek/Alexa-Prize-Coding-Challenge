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
labels = ['answer_question_external_fact', 'answer_question_recipe_steps',
       'ask_question_ingredients_tools', 'ask_question_recipe_steps',
       'ask_student_question', 'chitchat', 'misc', 'request_next_step',
       'return_list_ingredients_tools', 'return_next_step', 'stop']

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        'status' : 'alive'
    })

# start the flask app, allow remote connections 
app.run(host='0.0.0.0')

