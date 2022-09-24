import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error
from sklearn.datasets import make_regression
from plotly import graph_objs as go

def app():
    st.title("Cryptocurrency Predictor")

    def load_data(dataset_name):
   
        if dataset_name == "BTC":
            data = pd.read_csv('BTC.csv')
        elif dataset_name == "ETH":
            data = pd.read_csv('ETH.csv')
        elif dataset_name == "ETC":
            data = pd.read_csv('ETC.csv')
        elif dataset_name == "DOGE":
            data = pd.read_csv('DOGE.csv')
        elif dataset_name == "BCH":
            data = pd.read_csv('BCH.csv')
        return data
    
    #LINEAR REGRESSION

    def lLinearRegression(Closing_Predictor,Closing_to_Predict,Predictor_Closing_Price):
        x = Closing_Predictor['NextDay_Closing_Price'].values.reshape(-1,1)
        y = Closing_to_Predict['NextDay_Closing_Price'].values.reshape(-1,1)

        #Splitting the dataset
        train_X, test_X, train_y, test_y = train_test_split(x, y,random_state=0, test_size=.30)

        #Initialize the model
        model = LinearRegression()
        model.fit(train_X, train_y)

        #Make a specific prediction "%.4f"%
        y_prediction = model.predict([[Predictor_Closing_Price]])
        predicted_price = float(y_prediction)
        st.write("Predictor Closing Price\t\t" + str("%.4f"%Predictor_Closing_Price))
        st.write("Predicted Closing Price\t\t"+str("%.4f"%predicted_price))

        st.write("__________________________________________________")

        #Check Accuracy
        y_prediction = model.predict(test_X)
        MSE = mean_squared_error(test_y, y_prediction)
        MAE = mean_absolute_error(test_y, y_prediction)
        RMSE = np.sqrt(MSE)

        intecept = float(model.intercept_)
        slope = float(model.coef_)
        st.write("Model Intercept\t\t\t" + str("%.4f"%intecept))
        st.write("Model Slope\t\t\t" + str("%.4f"%slope))
        st.write("\nMean Absolute Error\t\t" + str("%.4f"%MAE))
        st.write("Root Mean Squared Error\t\t" + str("%.4f"%RMSE))
        st.write("R-Squared:")
        st.write("\tTraining Accuracy\t" + str("%.4f"%model.score(train_X, train_y)))
        st.write("\tTesting Accuracy\t" + str("%.4f"%model.score(test_X, test_y)))

        st.write("__________________________________________________")

        Result = pd.DataFrame({'Actual Price': test_y.flatten(), 'Predicted Price': y_prediction.flatten()})
        st.write(Result)

        xlabel = str(Closing_Predictor.iat[0,0])
        ylabel = str(Closing_to_Predict.iat[0,0])
        plt.scatter(test_X, test_y,  color='gray', s=0.5)
        plt.plot(test_X, y_prediction, color='red', linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        st.pyplot(plt)
        plt.clf()       
         
        df1 = Result.head(50)
        df1.plot(kind='bar',figsize=(16,10) )
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        st.pyplot(plt)
        plt.clf()

        ax = plt.gca()
        xlabel = str(Closing_Predictor.iat[0,0]) + ' and ' + str(Closing_to_Predict.iat[0,0])
        ylabel = str(Closing_to_Predict.iat[0,0]) + ' Closing Price'
        Result.plot(  y = 'Predicted Price', ax = ax, title = xlabel, ylabel = ylabel,lw=0.5)
        Result.plot( y = 'Actual Price' , ax = ax , lw=0.8)
        st.pyplot(plt)
        plt.clf()



    st.title("Linear Regression")
    form2 = st.form(key='my-from3')
    CP = form2.selectbox("Select Dataset of Closing Predictor", ("BTC","BCH","ETH","ETC","DOGE") ,key = "1")
    CP2 = form2.selectbox("Select Dataset of Closing to Predictor", ("BTC","BCH","ETH","ETC","DOGE"),key = "2")
    CPdata = load_data(CP)
    CP2data = load_data(CP2)
    pred = form2.text_input("Predictor Closing Price")
    submit = form2.form_submit_button('Predict')
    if submit:
        st.subheader('Raw data of Closing Predictor') 
        st.write(CPdata.tail()) 
        st.subheader('Raw data of Closing to Predictor') 
        st.write(CP2data.tail()) 
        lLinearRegression(CPdata,CP2data, float(pred))

