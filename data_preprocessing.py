#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[88]:


import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# # Importing the dataset

# In[89]:


def get_data(store_data, train_data, test_data):
    # import datasets
    store = pd.read_csv(
        store_data,
        dtype={
            "Store": "Int64",
            "StoreType": "object",
            "Assortment": "object",
            "CompetitionDistance" : "Int64",
            "CompetitionOpenSinceMonth": "Int64",
            "CompetitionOpenSinceYear": "Int64",
            "Promo2": "Int64",
            "Promo2SinceWeek": "Int64",
            "Promo2SinceYear": "Int64",
            "PromoInterval": "object",
            "StateHoliday": "object"
        })
    train = pd.read_csv(
        train_data,
        dtype={
            "Store": "Int64",
            "DayOfWeek": "Int64",
            "Date": "object",
            "Sales": "Int64",
            "Customers": "Int64",
            "StateHoliday": "object",
            "SchoolHoliday": "Int64"
        })
    test = pd.read_csv(
        test_data,
        dtype={
            "Id": "Int64",
            "Store": "Int64",
            "DayOfWeek": "Int64",
            "Date": "object",
            "Open": "Int64",
            "Promo": "Int64",
            "StateHoliday": "object",
            "SchoolHoliday": "Int64"
        })
    
    return store, train, test


# In[90]:


def merge_store(store, train, test):
    # merge the store_train and store_test dataset into the train and test dataset
    train = pd.merge(left=store, right=train, how='inner', left_on='Store', right_on='Store')
    test = pd.merge(left=store, right=test, how='inner', left_on='Store', right_on='Store')
    
    return train, test


# # Treat types and column names

# In[91]:


def split_date(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['WeekOfYear'] = data.Date.dt.isocalendar().week
    data = data.drop(['Date'],axis=1)
    
    return data


# In[92]:


# drop date column
def treat_data(data):
    data = data.rename(columns={"Store": "StoreId"})
    data = split_date(data)
    data['StateHoliday'] = data['StateHoliday'].apply(lambda x: 0 if x.isnumeric() else ord(x) - 64)
    data['CompetitionOpenSinceMonth'] = pd.to_datetime(data['CompetitionOpenSinceMonth']).dt.month
    data['CompetitionOpenSinceYear'] = pd.to_datetime(data['CompetitionOpenSinceYear']).dt.year
    data['Promo2SinceYear'] = pd.to_datetime(data['Promo2SinceYear']).dt.year
    data['DayOfWeek'] = pd.to_datetime(data['DayOfWeek']).dt.day
    
    return data


# # Split into feature and label

# In[93]:


def split_feature_label(train, label_column_name):
    X_train = train.drop([label_column_name], axis = 1)
    y_train = train[label_column_name]
    
    return X_train, y_train


# # Encode categorical values

# In[94]:


def encode_dummy(data, column_name):
    # use dummie variable to replace the PromoInterval column
    supported_columns = data[column_name].str.get_dummies(',').columns
    data_categories = data[column_name].str.get_dummies(',').filter(supported_columns)
    data = data.drop([column_name],axis = 1).join(data_categories)

    return data


# In[95]:


def encode_label(data, column_name):
    le = LabelEncoder()
    data[column_name] = le.fit_transform(data[column_name])
    
    pickle.dump(le, open('deploy/label_encoder.pkl', 'wb'))
    
    return data


# In[96]:


def encode_one_hot(data, column_name):
    encoder=OneHotEncoder(sparse=False)
    data_encoded = pd.DataFrame (encoder.fit_transform(data[[column_name]]))
    data_encoded.columns = encoder.get_feature_names([column_name])
    data.drop([column_name] ,axis=1, inplace=True)
    data= pd.concat([data, data_encoded ], axis=1)
    
    return data


# # Treat missing data

# In[97]:


def check_missing_data(data):
    return data.isnull().sum()


# In[98]:


def replace_numerical_mean(data):
    # replace all missing values by the mean in the column
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(data)
    output = imputer.transform(data)
    data = pd.DataFrame(output, columns = data.columns)
    
    return data


# # Export csv of treated data

# In[99]:


def convert_float_to_int(data):
    for x in data.columns.tolist():
        data[x] = np.floor(pd.to_numeric(data[x], errors='coerce')).astype('Int64')
    
    return data


# In[100]:


def export_csv(X_train, y_train, X_test):
    data = pd.concat([X_train, y_train ], axis=1)
    
    X_test = X_test.drop(columns=['Id'],axis=1)
    data = data.drop(columns=['Customers'],axis=1)
    
    data.to_csv('preprocessed_data/data.csv')
    X_test.to_csv('preprocessed_data/X_pred.csv')
    
    return None


# # Find last date with known sales

# # ETL

# In[101]:


if __name__ == '__main__':
    # load
    store_data = 'original_data/store.csv'
    train_data = 'original_data/train.csv'
    test_data = 'original_data/test.csv'
    store, train, test = get_data(store_data, train_data, test_data)

    # merge
    train, test = merge_store(store, train, test)

    # treat
    train = treat_data(train)
    test =  treat_data(test)

    # feature and label
    X_train, y_train = split_feature_label(train, 'Sales')
    X_test = test

    # encoding
    # encode dummy varibles for column with multiple values
    X_train = encode_dummy(X_train, 'PromoInterval')
    X_test = encode_dummy(X_test, 'PromoInterval')
    # encode label
    X_train = encode_label(X_train, 'Assortment')
    X_test = encode_label(X_test, 'Assortment')
    # encode one hot
    X_train = encode_one_hot(X_train, 'StoreType')
    X_test = encode_one_hot(X_test, 'StoreType')

    # missing data
    X_train = replace_numerical_mean(X_train)
    X_test = replace_numerical_mean(X_test)

    # convert data types before exporting
    y_train = pd.DataFrame(data=y_train)
    X_train = convert_float_to_int(X_train)
    X_test = convert_float_to_int(X_test)

    # export preprocessed data
    export_csv(X_train, y_train, X_test)


# In[ ]:




