import streamlit as st
import pickle
import numpy as np


st.set_page_config(
    page_title="Prediction App",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

from PIL import Image
image = Image.open('banner.jpg')

st.image(image,
      use_column_width=True)


def predict_age(Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight):
    input=np.array([[Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight]]).astype(np.float64)
    prediction = "1"
    return int(prediction)


def main():
    #st.title("Abalone Age Prediction")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Predição de crédito ao cliente </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)


    ID_Cliente = st.text_input("ID do Cliente")
    maiorAtraso = st.text_input("Maior atraso")
    margemBrutaAcumulada = st.text_input("Margem Bruta Acm.")
    percentualProtestos = st.text_input("Percentual de protestos (%)")
    prazoMedioRecebimentoVendas = st.text_input("Prazo médio de recebimento")
    titulosEmAberto = st.text_input("Títulos em Aberto")
    valorSolicitado = st.text_input("Valor de crédito solicitado")
    percentualRisco = st.text_input("Percentual de Risco")
    ativoCirculante = st.text_input("Ativo Circulante")
    passivoCirculante = st.text_input("Passivo Circulante")
    totalAtivo = st.text_input("Ativo Total")
    totalPatrimonioLiquido = st.text_input("Patrimonio Liquido Total")
    endividamento = st.text_input("Endividamento total")
    duplicatasAReceber = st.text_input("Duplicatas a Receber")
    estoque = st.text_input("Estoque declarado")
    faturamentoBruto = st.text_input("Faturamento Bruto")
    margemBruta = st.text_input("Margem Bruta")
    custos = st.text_input("Custos")
    capitalSocial = st.text_input("Capital Social")
    scorePontualidade = st.text_input("Score de Pontualidade")
    idadeNaSolicitacao = st.text_input("Idade da empresa")
    primeiraCompra_Anos = st.text_input("Ano da primeira compra")
    restricoes_True = st.text_input("Possui restrições? (1 = Sim | 0 = Não)")
    empresa_MeEppMei_True = st.text_input("É ME EPP Mei? (1 = Sim | 0 = Não)")


    warn_html ="""  
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
        output = predict_age(Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight)
        st.success('O crédito concedido ao cliente é de: R$ {}'.format(output))

        if output == 0:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(warn_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()