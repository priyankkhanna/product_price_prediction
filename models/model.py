###########################libraries import####################################3
from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
from tensorflow.keras.utils import pad_sequences
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

#############################33loading neccesseties###############
my_scaler = joblib.load('scaler.gz')
model = load_model('fasttext_rmlse_0.28_mae_782_mape_0.27.h5')
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

############taking input#################
x_title = input('title ')
x_description = input('description ')
cat1 = int(input('cat1 '))
cat2 = int(input('cat2 '))
cat3 = int(input('cat3 '))

#############processing input######################
def model_run(x_title,x_description,cat1,cat2,cat3):
    feed_data = {
        'x_title':[x_title],
        'x_description':[x_description],
        'category_1':[cat1],
        'category_2':[cat2],
        'category_3':[cat3]
    }
    dataset = pd.DataFrame(feed_data)
    dataset.x_title = dataset.x_title.str.replace('[^a-zA-Z0-9 ]', '')
    dataset.x_description = dataset.x_description.str.replace('[^a-zA-Z0-9 ]', '')
    max_title = 32
    max_description = 128
    dataset.x_title = tokenizer.texts_to_sequences(dataset.x_title)
    dataset.x_description = tokenizer.texts_to_sequences(dataset.x_description)

    #################converting input to feedable data################
    x_title = pad_sequences(dataset.x_title,maxlen = max_title)
    x_description = pad_sequences(dataset.x_description,maxlen = max_description)
    cat1 = dataset.category_1.to_numpy()
    cat2 = dataset.category_2.to_numpy()
    cat3 = dataset.category_3.to_numpy()
    feed_data = {
        'name':x_title,
        'item_desc':x_description,
        'category_1':cat1,
        'category_2':cat2,
        'category_3':cat3
    }
    #print(feed_data)
    #print(type(feed_data))
    #print(type(feed_data['name']))
    #####################generating results#######################
    y_pred = model.predict(feed_data)
    y_pred = my_scaler.inverse_transform(y_pred)
    y_pred = np.exp(y_pred)-1
    return y_pred
#############output#####################3
print(model_run(x_title,x_description,cat1,cat2,cat3))
