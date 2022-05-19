#Importing Needed Libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

#Setting Default Theme for plotly graphs
pio.templates.default = "simple_white"

#Reading data from github repository
df=pd.read_csv("https://raw.githubusercontent.com/sarard2/prediction/main/price.csv")

#Setting page width to wide
st.set_page_config(layout="wide")

# Sidebar Menu
with st.sidebar:
    selected = option_menu(menu_title=None,
    options=["Home","Data Overview","Exploratory Data Analysis","In-depth Analysis","Price Check","Ticket Issuing"],
    icons=["house","bar-chart","book","bell","bag-check","envelope"],
    menu_icon="cast",
    default_index=0,
    styles={
    "container": {"padding":"0!important"},"icon": {"color":"#grey"},
    "nav-link": {
    "font-size":"15px",
    "text-align":"left",
    "margin":"Opx","--hover-color":"#eee"},
    "nav-link-selected": {"background-color": "#fed8b1"},
    },)

#Home Page
if selected =="Home":
    st.header("Sara's Booking")
    st.image("https://makeflycheap.in/wp-content/uploads/2020/02/dd914c6cca076f8cebb463a81e73e7e5.jpg")
    st.subheader("Book Your Tickets Across India with the Best Prices!")
#Data Overview Page
if selected=="Data Overview":
    uploaded_file=st.file_uploader('',type=["csv"])
    if uploaded_file is not None:
        @st.cache
        def load_csv():
            csv = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1')
            return csv
            df = load_csv()
    st.header('Input DataFrame')
    st.write(df)
    st.write('---')



    with st.expander('Know more about the data!'):
     st.write("""
         The dataframe above shows a flight booking dataset collected from "Ease My Trip" website.  \nline
         The flight travel data is between the top 6 metro cities in India.  \nline
         It includes 300261 datapoints and 11 features.  \nline
         Duration: A continuous feature that displays the overall amount of time it takes to travel between cities in hours.  \nline
         Days Left: This is a derived characteristic that is calculated by subtracting the trip date by the booking date.  \nline
     """)

#Exploratory Page
if selected =="Exploratory Data Analysis":

    #To set each KPI in different column
    col1,col2,col3= st.columns(3)
    #KPIs
    col1.metric("Number of Studied Flights", "300261")
    col2.metric("Highest Ticket Price", "123071 INR")
    col3.metric("Unique Airlines","6")

    #To count the number of flights per Airline
    dff=df.groupby("Airline").count()
    df_groupby=dff.reset_index()
    figure6 = px.bar(df_groupby, x="Airline",y="SourceCity",color='Price',color_continuous_scale='RdBu',title="Amount of Flights per Airline")
    figure6.update_layout(xaxis_title=None,yaxis_title=None)
    figure6.update_xaxes(showgrid=False,zeroline=False)
    figure6.update_yaxes(showgrid=False,showticklabels = True)
    st.plotly_chart(figure6,use_container_width=True)

    #To check the dsitribution of flights between classes and airlines
    figure2 = px.treemap(df, path=['Airline', 'Class'], values=df.nunique(axis=1),
                              color='Price',
                              color_continuous_scale='RdBu',title="Ticket Type per Airline")
    figure2.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(figure2,use_container_width=True)

#Analysis Page
if selected=="In-depth Analysis":
    col1, col2 = st.columns(2)
    figure3, ax = plt.subplots()

    #To check the prices across airlines
    figure3=sns.catplot(y = "Price", x = "Airline", data = df.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3,ax=ax)
    st.pyplot(figure3)
    with col1:
    #To check the prices across destination cities
        figure5, axx = plt.subplots()
        figure5=sns.catplot(y = "Price", x = "DestinationCity", data = df.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3,ax=axx)
        st.pyplot(figure5)

    with col2:
    #To check the prices across source cities
        figure4, axx = plt.subplots()
        figure4=sns.catplot(y = "Price", x = "SourceCity", data = df.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3,ax=axx)
        st.pyplot(figure4)

#ML Page
if selected =="Price Check":
#Linear Regression Model to predict approximate price of tickets based on variables that the user inputs
   def main():
    #Source City
       col1, col2,col3 = st.columns([1,1,2])
       with col1:
           st.subheader("Source City")
           source = st.selectbox("from India" , ['Bangalore', 'Chennai','Delhi','Hyderabad','Kolkata',"Mumbai"])
           st.write("You are traveling from  " , source)
       if source == "Bangalore":
           source_inp = 0
       elif source == "Chennai":
           source_inp = 1
       elif source == "Delhi":
           source_inp = 2
       elif source == "Hyderabad":
           source_inp = 3
       elif source == "Kolkata":
           source_inp = 4
       elif source == "Mumbai":
           source_inp = 5

     #Destination City
       with col2:
           st.subheader("Destination City")
           dest = st.selectbox("to India" , ['Bangalore', "Chennai",'Delhi', 'Hyderabad','Kolkata','Mumbai'])
           st.write("You are traveling towards ",dest)
       if dest == "Bangalore":
           dest_inp = 0
       elif dest == "Chennai":
           dest_inp = 1
       elif dest == "Hyderabad":
           dest_inp = 2
       elif dest == "Delhi":
           dest_inp = 3
       elif dest == "Kolkata":
           dest_inp = 4
       elif dest == "Mumbai":
           dest_inp = 5

       #Airline
       with col3:
           st.subheader("Select Airline")
           airline = st.selectbox("" , ["AirAsia","GOFIRST","Air India","Indigo","SpiceJet","Vistara"])
           st.write("You chose " , airline," Airline")
       if airline == "AirAsia":
           air_inp = 1
       elif airline == "GOFIRST":
           air_inp = 2
       elif airline == "Air India":
           air_inp = 3
       elif airline == "Indigo":
           air_inp = 4
       elif airline == "SpiceJet":
           air_inp = 5
       elif airline == "Vistara":
           air_inp = 6

       #Arrival Time (denoted as a time interval)
       with col2:
           st.subheader("Arrival Time")
           arrival = st.selectbox(" " , ["Afternoon","EarlyMorning","Evening","LateNight","Morning","Night"])
           st.write("You will arrive in the " , arrival)
       if arrival == "Afternoon":
           arrival_inp = 0
       elif airline == "EarlyMorning":
           arrival_inp = 1
       elif airline == "Evening":
           arrival_inp = 2
       elif airline == "LateNight":
           arrival_inp = 3
       elif airline == "Morning":
           arrival_inp = 4
       elif airline == "Night":
           arrival_inp = 5
       else:
           arrival_inp=6


       #Departure Time (denoted as a time interval)
       with col1:
           st.subheader("Departure Time")
           departure = st.selectbox("  " , ["Afternoon","EarlyMorning","Evening","LateNight","Morning","Night"])
           st.write("You will depart in the " , departure)
       if departure == "Afternoon":
           dep_inp = 0
       elif departure == "EarlyMorning":
           dep_inp = 1
       elif departure == "Evening":
           dep_inp = 2
       elif departure == "LateNight":
           dep_inp = 3
       elif departure == "Morning":
           dep_inp = 4
       elif departure == "Night":
           dep_inp = 5
       else:
           dep_inp=6


       #Class of ticket (Economy or Business)
       with col3:
           st.subheader("Select the Class of your Ticket")
           ticketclass= st.selectbox("Class" , ["Economy","Business"])
           st.write("Your ticket class type is  " , ticketclass)
       if ticketclass == "Economy":
           class_inp = 0
       elif ticketclass == "Business":
           class_inp = 1


       #Number of Stops
       with col3:
           st.subheader("Number of Stops")
           stop = st.selectbox("if any" , [0,1,2])
           st.write("You will have ", stop, "stops")

       #Duration of Flight
       with col1:
           st.subheader("Check Duration")
           dur=st.number_input("in hours")
           st.write("Your trip will take ", dur)
       #Days Until Flight
       with col2:
           st.subheader("Days Until Flight")
           day=st.number_input("    ")
           st.write("You still have ", day, "days until your flight")

    #Model of Linear Regression From Pickle File
       lrmodel=pickle.load(open("https://raw.githubusercontent.com/sarard2/prediction/main/linearmodel.pkl",'rb'))
    #Prediction given variables
       par = [air_inp , source_inp , dest_inp ,stop, class_inp , arrival_inp, dep_inp, dur, day]

       if st.checkbox("Check Your Ticket Price"):
           pred = lrmodel.predict([par])
           for i in pred:
               st.write("Your Fare Price is : " , round(i ,3)  , "INR")
               st.write("*Happy and Safe Journey ...*")

       st.write("""    """)
       st.write("""    """)

   if __name__ == "__main__":
       main()
if selected =="Ticket Issuing":
    st.write("hiii")
