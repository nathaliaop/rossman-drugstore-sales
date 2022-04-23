import os
import pandas as pd
import pickle
from flask import Flask, request

from api_preprocessing import ApiPreprocessing

# import model
regressor = pickle.load(open('deploy/random_forest_sales_prediction.pkl', 'rb'))

# load dataset
dataset = pd.read_csv('preprocessed_data/X_pred.csv')

# instanciate flask
app = Flask(__name__)

# route for predictions
@app.route('/predict', methods=['POST'])
def predict():
  test_json = request.get_json()

  # get the data requested
  df_raw = dataset[(dataset['Year'] == int(test_json['Year'])) & (dataset['Month'] == int(test_json['Month']))]

  # instanciate data preparation
  pipeline = ApiPreprocessing()

  # data preparation
  df1 = pipeline.data_preparation(df_raw)

  # predict
  pred = regressor.predict(df1)

  # create prediction column
  df_raw['Prediction'] = pred

  # return dataframe
  return df_raw.to_json(orient='records')

if __name__ == '__main__':
  # start flask
  port = os.environ.get('PORT', 5000)
  app.run(host='0.0.0.0', port=port)