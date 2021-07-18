import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ANN:
    def __init__(self,selected_features_index,data):
        self.selected_features_index = selected_features_index
        self.data = data

    def mape_calc(actual, pred): 
        actual, pred = np.array(actual), np.array(pred)
        return np.mean(np.abs((actual - pred) / actual)) * 100

    def smape(a, f):
        return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

    def test_model():
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

        X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size=0.3, random_state=42)

        tf.random.set_seed(42)

        model = tf.keras.Sequential([
                                    tf.keras.layers.Dense(100,activation='relu'),
                                    tf.keras.layers.Dense(100,activation='relu'),
                                    tf.keras.layers.Dense(100,activation='relu'),
                                    tf.keras.layers.Dense(10),
                                    tf.keras.layers.Dense(1)

        ])

        model.compile(loss=tf.keras.losses.mape,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        metrics=['mape'])

        history = model.fit(X_train,y_train, epochs=355,verbose=0)

        loss, mape = model.evaluate(X_test,y_test)
        y_pred = model.predict(X_test)
        y_pred = np.asarray(y_pred).reshape(480,1)
        y_test = np.asarray(y_test).reshape(480,1)
        test_error = mape_calc(y_test,y_pred)
        return test_error

