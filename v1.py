# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:57:10 2022

@author: UrsHu
"""
#########################################################

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

#########################################################

st.set_page_config(
    page_title= "Customer Prediciton App",
    page_icon = "📈",
    layout="wide")
  
  st.markdown("This app is meant as a tool to predict income of a set of customer. The dataset used was found on" +
            " [Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)\
                .")
#########################################################

@st.cache()
def load_data():
    data = pd.read_csv("Data1.csv")
    return(data.dropna())

@st.cache(allow_output_mutation=True)
def load_model1():
    filename = "finalized_income_model1.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return(loaded_model)

def load_model2():
    filename = "finalized_extended_income_model.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return(loaded_model)

data = load_data()
model1 = load_model1()
model2 = load_model2()

#########################################################

st.title("Customer Prediction App")
st.markdown("This application is a Streamlit dashboard that can be used to *analyze* and **predict** customer income")

#########################################################
st.header(" ")
st.header(" ")
    
row1_col1, row1_col2, row1_col3 = st. columns([1,1,1])

#Slider1
wine = row1_col1.slider(label="Wine", min_value = 0,
                        max_value = 500, value=(312),
                        step = None)
#Slider2
fruits = row1_col2.slider("Fruits",
                  0,
                  500,
                  value=(27))
#Slider3
fish = row1_col3.slider("Fish Products",
                  0,
                  500,
                  value=(38))

row2_col1, row2_col2, row2_col3 = st.columns([1,1,1])

#Slider4
meat = row2_col1.slider("Meat Products",
                  0,
                  500,
                  value=(170))
#Slider5
sweet = row2_col2.slider("Sweet Products",
                  0,
                  500,
                  value=(28))
#Slider6
gold = row2_col3.slider("Gold Products",
                  0,
                  500,
                  value=(44))

data = {"MntWines":  [wine],
        'MntFruits': [fruits],
        'MntFishProducts':  [fish],
        'MntMeatProducts': [meat],
        'MntSweetProducts':  [sweet],
        'MntGoldProds': [gold]}

labels = ["MntWines","MntFruits","MntFishProducts","MntMeatProducts","MntSweetProducts","MntGoldProds"]
values = [wine , fruits , fish , meat , sweet , gold]
colors1 = {"coral", "lightcoral", "orangered", "darkred", "tomato", "crimson"}


df = pd.DataFrame(data)

Income = model1.predict(df)
difference = round(Income[0] - 55000, 2)
col1, col2 = st.columns(2)


with col1:
    st.header(" ")
    st.header(" ")
    
    fig1, ax1 = plt.subplots(figsize=(5,5))
    wedges, texts = ax1.pie(values, wedgeprops=dict(width=0.5), startangle=-40, colors = colors1)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax1.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

    col1.pyplot(fig1, use_container_width=True)

with col2:
    st.header("Customer Purchases")
    st.dataframe(df)
    st.metric("Predicted Customer Income", Income, delta = round(difference, 2), delta_color="normal")
    st.text("ABOVE/BELOW average of approximatly 55 000 USD")
####################################################
st.header(" ")
st.header(" ")

st.header("Predicting Customer Income with more Data")
uploaded_data = st.file_uploader("Choose a file with Customer Data for Predicting Customer Income")

# Add action to be done if file is uploaded
if uploaded_data is not None:
    
    # Getting Data and Making Predictions
    guinea_pigs = pd.read_csv(uploaded_data)
    guinea_pigs["predicted_income"] = model2.predict(guinea_pigs)
    
    # Add User Feedback
    st.success("🕺🏽🎉👍 You successfully scored %i new pigs for Income prediction! 🕺🏽🎉👍" % guinea_pigs.shape[0])
    
    # Add Download Button
    st.download_button(label = "Download scored customer data",
                       data = guinea_pigs.to_csv().encode("utf-8"),
                 
                       file_name = "scored_customer_data.csv")
    st.dataframe(guinea_pigs)

    cola, colb, colc, cold, cole, colf, colg, colh = st.columns(8)
    with cola:
        awp = guinea_pigs["MntWines"].mean()
        st.metric("Average Wine Purchases", round(awp,2))
    with colb:
        afp = guinea_pigs["MntFruits"].mean()
        st.metric("Average Fruit Purchases", round(afp,2))
    with colc:
        amp = guinea_pigs["MntMeatProducts"].mean()
        st.metric("Average Meat Purchases", round(amp,2))
    with cold:
        afip = guinea_pigs["MntFishProducts"].mean()
        st.metric("Average Fish Purchases", round(afip,2))
    with cole:
        asp = guinea_pigs["MntSweetProducts"].mean()
        st.metric("Average Sweet Purchases", round(asp,2))
    with colf:
        agp = guinea_pigs["MntGoldProds"].mean()
        st.metric("Average Gold Purchases", round(agp,2))
    with colg:
        aa = guinea_pigs["Age"].mean()
        st.metric("Average Age", round(aa,0))
    with colh:
        ap = guinea_pigs["predicted_income"].mean()
        st.metric("Average Income",  round(ap,2))
     

    colaa, colbb = st.columns(2)

    with colaa:
        st.header("Aggregated Shoppingbasket")
        st.markdown("Based on the newly imported data")
        st.header(" ")
        st.text(" ")
    
        values2 = [awp,afp,amp,afip,asp,agp]
    

        fig, axs = plt.subplots(figsize=(5,5))
        wedges, texts = axs.pie(values2, wedgeprops=dict(width=0.5), startangle=-40, colors = colors1)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        axs.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

    colaa.pyplot(fig, use_container_width=True)

    with colbb:
        st.header("Purchases in comparison to predicted income")
    
        options = st.selectbox(
            'Which purchases would you like to compare?',
            ('MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',"MntSweetProducts","MntGoldProds"))
    
        guinea_pigs.sort_values(by=[options])
    
    
    fig3, axs = plt.subplots(figsize=(10,6.5))
    axs.scatter(guinea_pigs["predicted_income"], guinea_pigs[options], color = "crimson")
    axs.set_ylabel(options + " in Units")
    axs.set_xlabel("Predicted Income in USD")
    colbb.pyplot(fig3, use_container_width=True)


else :
    st.warning("⚠️Please upload Guineapigs file to continue⚠️", ) 









