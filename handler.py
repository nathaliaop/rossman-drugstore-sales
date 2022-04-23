import os
import pandas as pd
import pickle
from flask import Flask, request

from api_preprocessing import ApiPreprocessing

# import model
regressor = pickle.load(open('deploy/random_forest_sales_prediction.pkl', 'rb'))

# instanciate flask
app = Flask(__name__)

# route for predictions
@app.route('/predict', methods=['POST'])
def predict():
  test_json = request.get_json()

  # if there's data in the request, convert the request to a dataframe
  if test_json:
    if isinstance(test_json, dict): # unique value
      df_raw = pd.DataFrame(test_json, index=[0])
    else:
      df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

  # instanciate data preparation
  pipeline = ApiPreprocessing()

  # data preparation
  df1 = pipeline.data_preparation(df_raw)

  # predict
  pred = regressor.predict(df1)

  # create prediction column
  df_raw['prediction'] = pred

  # return dataframe
  return df_raw.to_json(orient='records')

if __name__ == '__main__':
  # start flask
  port = os.environ.get('PORT', 5000)
  app.run(host='0.0.0.0', port=port)