import pickle
file_name = "xgb_reg.pkl"
xgb_aprovado = pickle.load(open('Modelos\\xgb_aprovado.pkl', "rb"))
xgb_credito = pickle.load(open('Modelos\\xgb_credito.pkl', "rb"))
print("OK")