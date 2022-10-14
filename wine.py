import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 
import seaborn as sns

data = pd.read_csv("data//wine.csv")

tdata = data

lin_regression = LinearRegression()
#Drop year and population
data.drop(['Year','FrancePop'], axis = 1, inplace = True)
#Create Independent and Dependent Variables
X = data.drop('Price', axis = 1)
y = data['Price']
lrMod = lin_regression.fit(X, y)


st.title("The Artificial Sommelier")
st.image("data//wine.jpg",width = 800)

nav = st.sidebar.radio("Navigation",["Home", "Prediction", "Contribute"])
if nav == 'Home':
    st.write("home")
    if st.checkbox("Show Data"):
        st.table(tdata)

    graph = st.selectbox("What kind of Graph?", ["Non-Interactive", "Interactive"])
    if graph == "Non-Interactive":
        figure, axis = plt.subplots(2, 2, sharey='row')
        sns.regplot(data = data,
                x = 'AGST',
                y = 'Price',
                ax=axis[0, 0],
                ci = None)
        sns.regplot(data = data,
                x = 'Age',
                y = 'Price',
                ax=axis[0, 1],
                ci = None)
        sns.regplot(data = data,
                x = 'WinterRain',
                y = 'Price',
                ax=axis[1, 0],
                ci = None)
        sns.regplot(data = data,
                x = 'HarvestRain',
                y = 'Price',
                ax=axis[1, 1],
                ci = None)
        figure.tight_layout()
        st.pyplot(figure)

    if graph == "Interactive":
        pvar = st.selectbox("Price Vs ?", ["AGST", "Age","WinterRain","HarvestRain"])
        if pvar == "Age":
            val = st.slider("Filter data using years",0,30)
            fdata = data.loc[data["Age"]>= val]
        else:
            fdata = data
        fig = go.Figure(data=go.Scatter(x=fdata[pvar], y=fdata["Price"], mode='markers'))
        st.plotly_chart(fig)

if nav == "Prediction":
    st.header("Predict the Wine Price")
    v_age = st.number_input("Enter the age of the wine:",0.00, 30.00, step = 1.00, value= 5.00)
    v_agst = st.number_input("Enter the Average Growing Season Temperature:",15.00, 18.00, step = 0.20, value= 15.00)
    v_wr = st.number_input("Enter the Winter Rain:",300.00, 1000.00, step = 10.00, value= 300.00)
    v_hr = st.number_input("Enter the Harvest Rain:",50.00, 300.00, step = 10.00, value= 50.00)
    
    test_data = pd.DataFrame(
        dict(WinterRain = v_wr,
             AGST = v_agst,
             HarvestRain = v_hr,
             Age = v_age),
        index=[0]
    )
    #st.write(test_data)
    pred = lrMod.predict(test_data)

    if st.button("Predict"):
        st.success(f"Your predicted wine price is £ {round (pred[0],2)}")

if nav == "Contribute":
    st.header("Contribute to our dataset")
    v_age = st.number_input("Enter the age of the wine:",0.00, 30.00, step = 1.00, value= 5.00)
    v_agst = st.number_input("Enter the Average Growing Season Temperature:",15.00, 18.00, step = 0.20, value= 15.00)
    v_wr = st.number_input("Enter the Winter Rain:",300.00, 1000.00, step = 10.00, value= 300.00)
    v_hr = st.number_input("Enter the Harvest Rain:",50.00, 300.00, step = 10.00, value= 50.00)
    v_year = st.number_input("Enter the Year:",1980.00, 2022.00, step = 1.00, value= 1980.00)
    v_fpop = st.number_input("Enter the French Population:",54602.193, 100000.00, step = 1000.00, value= 54602.193)
    v_pr = st.number_input("Enter the Price:",4.00, 10.00, step = 0.10, value= 5.00)
    
    data2add = pd.DataFrame(
        dict(Year = v_year,
             Price = v_pr,
             WinterRain = v_wr,
             AGST = v_agst,
             HarvestRain = v_hr,
             Age = v_age,
             FrancePop = v_fpop),
        index=[0]
    )

    if st.button("submit"):
        data2add.to_csv("data//wine.csv",mode='a',header = False,index= False)
        st.success("Submitted")