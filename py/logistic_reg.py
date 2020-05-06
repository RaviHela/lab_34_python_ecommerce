
from sklearn import linear_model
import joblib

def lr_predict(X):
    
    lr_model = joblib.load("models/logistic_reg.sav")
    return lr_model.predict(X)
    
    
