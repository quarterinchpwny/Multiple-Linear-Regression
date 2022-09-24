import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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


    def MultipleLinearRegression(df, Opening_Price, Price_High, Price_Low):
        df = pd.DataFrame(df, columns=['Currency','Date','Opening_Price', '24h_High', '24h_Low', 'NextDay_Closing_Price'])
        X = df[['Opening_Price', '24h_High', '24h_Low']]
        y = df['NextDay_Closing_Price']

        #Splitting the dataset
        train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0, test_size=.30)

        #Initialize the model
        model = LinearRegression()
        model.fit(train_X, train_y)

        #Make a specific prediction
        y_prediction = model.predict([[Opening_Price, Price_High, Price_Low]])
        st.write("Opening Price\t\t\t" + str("%.4f"%Opening_Price))
        st.write("24h High\t\t\t"  + str("%.4f"%Price_High))
        st.write("24h Low\t\t\t\t"  + str("%.4f"%Price_Low))
        st.write("Predicted Closing Price\t\t"+str("%.4f"%y_prediction))
        st.write("__________________________________________________")

        #Check Accuracy
        y_prediction = model.predict(test_X)
        MSE = mean_squared_error(test_y, y_prediction)
        RMSE = np.sqrt(MSE)
        R2 = r2_score(test_y, y_prediction)
    
        #Put it in one dataframe
        Result = pd.DataFrame(data={'Predicted Price': y_prediction, 'Actual Price': test_y})
        dates = pd.Series([],dtype='object')
        for row1 in df.index:
         for row2 in Result.index:
                if (row1 == row2):
                    date = df["Date"].loc[row1]
                    dates[row2] = date

        Result.insert(0, "Date", dates)
        Result = Result.sort_index(axis=0,ascending=True)
    
        st.write("Model Intercept\t\t\t" + str("%.4f"%model.intercept_))
        st.write("Model Slope\t\t\t" + str(model.coef_))
        st.write("\nMean Squared Error\t\t" + str("%.4f"%MSE))
        st.write("Root Mean Squared Error\t\t" + str("%.4f"%RMSE))
        st.write("R-Squared:")
        st.write("\tTraining Accuracy\t" + str("%.4f"%model.score(train_X, train_y)))
        st.write("\tTesting Accuracy\t" + str("%.4f"%model.score(test_X, test_y)))
        st.write("__________________________________________________")

        st.write (Result)
    
        ax = plt.gca()
        ylabel = str(df.iat[0,0]) + ' Closing Price'
        #Result.plot( x = 'Date' , y = 'Predicted Price', ax = ax, ylabel = 'Closing Price', lw=0.5)
        #Result.plot( x = 'Date' , y = 'Actual Price' , ax = ax , lw=0.5)
        Result.plot( x = 'Date' , y = 'Predicted Price', ax = ax, ylabel = ylabel, lw=0.5)
        Result.plot( x = 'Date' , y = 'Actual Price' , ax = ax , lw=0.5)
    

        st.pyplot(plt)

        df1 = Result.head(50)
        df1.plot(kind='bar',figsize=(16,10))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        st.pyplot(plt)




    st.title("Multiple Linear Regression")

    form2 = st.form(key='my-from3')
    dataset_name = form2.selectbox("Select Dataset", ("BTC","BCH","ETH","ETC","DOGE"))
    data = load_data(dataset_name)
    OP = form2.text_input("Opening Price")
    PH = form2.text_input("24 Price High")
    PL = form2.text_input("24 Price Low")
    submit = form2.form_submit_button('Predict')
    if submit:
        st.subheader('Raw data') 
        st.write(data.tail()) 
        MultipleLinearRegression(data,float(OP), float(PH), float(PL))


















