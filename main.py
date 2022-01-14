#Import streamlit, yfinance, fbprophet, plotly
import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

#Start day 
start_day = "2017-01-01"
#End day(today)
end_day = date.today().strftime("%Y-%m-%d")

#Title
st.title("Stock Prediction")

#List of stock's company that we want to predict
company = ("AAPL", "GOOG", "MSFT", "TSLA","FB")
#Selected_box for company choices
selected_stock = st.selectbox("Select data set for prediction", company)
#Year of prediction
num_month = st.slider("Months of prediction (30 days): ", 1, 12)
#Period of prediction
period = num_month * 30

#Function to load data and cache data with st.cache
@st.cache
def get_data(ticker):
    data = yf.download(ticker, start_day, end_day)    #download data from start to today
    data.reset_index(inplace=True)              #get index of data
    return data[['Date', 'Open', 'Close', 'High', 'Low']]

#Load data
data_loading = st.text("Loading data ...")
data = get_data(selected_stock)
data_loading.text("Loading data ... Done!")

st.subheader("Raw Data Table")                  #header of data_loaded
st.write(data.tail())                           #show tail of data

#Function to plot data
def plot_raw_data():
    fig = go.Figure()                                                                   #Get graph figure
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="StockOpen"))         #First trace: x:Data, y:Open
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="StockClose"))       #Second trace: x:Data, y:Close
    #Title of the graph and create range slider for x axis
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)    
    st.plotly_chart(fig)                                                                #Plot

plot_raw_data()

#Prediction our data
data_frame_train = data[["Date", "Close"]]
data_frame_train = data_frame_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(data_frame_train)                                     #Fit the model
future = m.make_future_dataframe(periods=period)    #Get suitable future data frame
forecast = m.predict(future)                       

#Forecast data table
st.subheader("Forecast Data")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

#Plot the prediction data
st.write("Forecast Data Graph")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

#Plot the components
st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)