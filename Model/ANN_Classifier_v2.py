import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

tf.random.set_seed(42) #random seed

class ANN:
    
    def __init__(self,selected_features_index,data):
        self.selected_features_index = selected_features_index
        self.data = data
    
    def data_preprocessing(self):
        selected_features_index = self.selected_features_index
        data = self.data
        X = data.drop("target", axis = 1)
        y = data["target"]

        selected_features = []
        cols = X.columns

        for index in range(len(selected_features_index)):
            if selected_features_index[index] == 1:
                selected_features.append(cols[index])  

        for i in range(len(cols)):
            k = cols[i]
            if k not in selected_features:
                X = X.drop(columns = k)
            
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X) 
        X_train, X_rem, y_train, y_rem = train_test_split(X_scaled,y, test_size=0.3, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5, random_state=42)
        
        #saving for later use
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_rem
        self.y_test = y_rem

    def mape_calc(self, actual, pred): 
        actual, pred = np.array(actual), np.array(pred)
        return np.mean(np.abs((actual - pred) / actual)) * 100

    def smape(self, a, f):
        return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

    def build_model(self):
        self.model = tf.keras.Sequential([
                                    tf.keras.layers.Dense(50,activation='relu'),
                                    tf.keras.layers.Dense(30,activation='relu'),
                                    tf.keras.layers.Dense(30,activation='relu'),
                                    tf.keras.layers.Dense(30,activation='relu'),
                                    tf.keras.layers.Dense(20,activation='relu'),
                                    tf.keras.layers.Dense(20,activation='relu'),
                                    tf.keras.layers.Dense(10,activation='relu'),
                                    tf.keras.layers.Dense(10,activation='relu'),
                                    tf.keras.layers.Dense(5),
                                    tf.keras.layers.Dense(1)

        ])

        self.model.compile(loss=tf.keras.losses.mape,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        metrics=['mape'])
    
    def train(self):
        self.data_preprocessing()
        self.build_model()
        self.history = self.model.fit(self.X_train,self.y_train, epochs=30,verbose=0)
    
    def test_error(self):
        loss, mape = self.model.evaluate(self.X_test,self.y_test)
        y_pred = self.model.predict(self.X_test)

        y_pred = np.asarray(y_pred)
        y_test = np.asarray(self.y_test)

        test_error = self.mape_calc(y_test,y_pred)
        return test_error
    
    def validation_error(self):
        pass

