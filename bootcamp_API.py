import streamlit as st
import pickle
import numpy as np
import pandas as pd

df = pd.read_csv('Database/Base_Clientes.csv')

st.set_page_config(
    page_title="Prediction App",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

from PIL import Image
image = Image.open('banner.jpg')

st.image(image, use_column_width=True)


def predict_age(ID_Cliente):
    predict = df.loc[df["ID_Cliente"]==ID_Cliente]
    prediction = predict["Aprovado"]
    return prediction.values

def predict_minimo(ID_Cliente):
    predict = df.loc[df["ID_Cliente"]==ID_Cliente]
    prediction = predict["Minimo"]
    return prediction


def main():
    #st.title("Abalone Age Prediction")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Predição de crédito ao cliente </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)


    ID_Cliente = st.text_input("ID do Cliente")
    valorSolicitado = st.text_input("Valor de crédito solicitado")


    aprovado_html ="""  
      <div style="background-color:#80ff80; padding:10px >
      <h2 style="color:white;text-align:center;"> Cliente Aprovado</h2>
      </div>
    """
    reprovado_html="""  
      <div style="background-color:#F08080; padding:10px >
       <h2 style="color:black ;text-align:center;"> Cliente reprovado</h2>
       </div>
    """

    if st.button("Predição de crédito"):
        output = predict_age(ID_Cliente)
        minimo = predict_minimo(ID_Cliente)
        st.success('O crédito concedido ao cliente é: {}'.format(minimo))

        if output == "B":
            st.markdown(reprovado_html,unsafe_allow_html=True)
        else:
            st.markdown(aprovado_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()