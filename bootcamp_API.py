import streamlit as st
import pickle
import numpy as np

file_name = "xgb_reg.pkl"
xgb_aprovado = pickle.load(open('Modelos\\xgb_aprovado.pkl', "rb"))
xgb_credito = pickle.load(open('Modelos\\xgb_credito.pkl', "rb"))
print("OK")


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
