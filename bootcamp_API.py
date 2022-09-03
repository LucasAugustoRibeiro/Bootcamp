import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt


xgb_aprovado = pickle.load(open('xgb_aprovado.pkl', "rb"))
xgb_credito = pickle.load(open('Modelos\\xgb_credito.pkl', "rb"))



def predict_age(Length,Diameter,Height,Whole_weight,Shucked_weight,
                Viscera_weight,Shell_weight):
    input=np.array([[Length,Diameter,Height,Whole_weight,Shucked_weight,
                     Viscera_weight,Shell_weight]]).astype(np.float64)
    prediction = xgb_aprovado.predict(input)
    
    return int(prediction)


def main():
    st.title("Abalone Age Prediction")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Abalone Age Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    
    Length = st.text_input("Length","Type Here")
    Diameter = st.text_input("Diameter","Type Here")
    Height = st.text_input("Height","Type Here")
    Whole_weight = st.text_input("Whole weight","Type Here")
    Shucked_weight = st.text_input("Shucked weight","Type Here")
    Viscera_weight = st.text_input("Viscera weight","Type Here")
    Shell_weight = st.text_input("Shell weight","Type Here")

    safe_html ="""  
    <div style="background-color:#80ff80; padding:10px >
    <h2 style="color:white;text-align:center;"> The Abalone is young</h2>
    </div>
    """
    if st.button("Predict the age"):
    output = predict_age(Length,Diameter,Height,Whole_weight,
                        Shucked_weight,Viscera_weight,Shell_weight)
    st.success('The age is {}'.format(output))

    if output == 1:
        st.markdown(safe_html,unsafe_allow_html=True)
    elif output == 2:
        st.markdown(warn_html,unsafe_allow_html=True)
    elif output == 3:
        st.markdown(danger_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()