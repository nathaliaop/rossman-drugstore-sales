import pickle

class ApiPreprocessing(object):
  def __init__(self):
    self.label_encoder = pickle.load(open('deploy/label_encoder.pkl', 'rb'))

  def data_preparation(self, data):
    data['Assortment'] = self.label_encoder.fit_transform(data['Assortment'])

    return data