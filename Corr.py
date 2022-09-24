import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error
from sklearn.datasets import make_regression
from plotly import graph_objs as go


def app():
    BTC = pd.read_csv('BTC.csv')
    ETH = pd.read_csv('ETH.csv')
    BCH = pd.read_csv('BCH.csv')
    ETC = pd.read_csv('ETC.csv')
    DOGE = pd.read_csv('DOGE.csv')
    #Correlation Analysis

    #creating dataframes for each cryptocurrency
    BTC_Close = pd.DataFrame(BTC,columns=['Date','Currency','Today_Closing_Price'])
    ETH_Close = pd.DataFrame(ETH,columns=['Date','Currency','Today_Closing_Price'])
    BCH_Close = pd.DataFrame(BCH,columns=['Date','Currency','Today_Closing_Price'])
    ETC_Close = pd.DataFrame(ETC,columns=['Date','Currency','Today_Closing_Price'])
    DOGE_Close = pd.DataFrame(DOGE,columns=['Date','Currency','Today_Closing_Price'])

    #merging all dataframes and renaming columns
    BTC_ETH = pd.merge(BTC_Close,ETH_Close, on='Date',how='outer')
    temp = pd.merge(BCH_Close,ETC_Close, on='Date',how='outer')
    BCH_ETC_DOGE = pd.merge(temp,DOGE_Close, on='Date',how='outer')
    Cryptocurrencies = pd.merge(BTC_ETH,BCH_ETC_DOGE, on='Date',how='outer')
    Cryptocurrencies.rename(columns={'Currency_x_x':'BTC','Today_Closing_Price_x_x':'BTC_Price','Currency_y_x':'ETH','Today_Closing_Price_y_x':'ETH_Price',
                      'Currency_x_y':'BCH','Today_Closing_Price_x_y':'BCH_Price','Currency_y_y':'ETC','Today_Closing_Price_y_y':'ETC_Price',
                      'Currency':'DOGE','Today_Closing_Price':'DOGE_Price'}, inplace=True)
                      
    print(Cryptocurrencies.head(5))

    #Pearson Correlation Analysis
    Correlation_Analysis = Cryptocurrencies.corr(method="pearson")

    col1, col2 = st.columns(2)
    st.write("Pearson correlation coefficient")
    with col1:
        st.write("Bitcoin (BTC):")
        st.write(Correlation_Analysis["BTC_Price"])
    with col2:
        st.write("Ethereum (ETH):")
        st.write(Correlation_Analysis["ETH_Price"])

    sn.heatmap(Correlation_Analysis, annot=True)
    st.pyplot(plt)