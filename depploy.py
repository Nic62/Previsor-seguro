import joblib as jb
import streamlit as st
import pandas as pd

#.\ambv\Scripts\activate
#streamlit run depploy.py
# Carregar o modelo treinado
#pip freeze > requirements.txt


modelo = jb.load("bestxgb.pkl")
scaler = jb.load("scaler.pkl")

# Título e descrição
st.title("Seguro de Saúde - Previsão")
st.divider()
st.header('Digite as informações abaixo:')

# Entrada dos dados pelo usuário
info_idade = st.number_input("Digite sua idade",min_value=18,max_value=100)
info_sexo = st.selectbox("Escolha seu sexo", ['Masculino', 'Feminino', 'Prefiro não dizer'])
info_fumante = st.radio("Você é fumante?", ['Sim', 'Não'])
info_criança = st.select_slider("Selecione o número de filhos", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"])

if info_criança == "10+":
    total_filho = st.number_input('Se maior que dez, quantos:', min_value=10)
else:
    total_filho = int(info_criança)

info_regiao = st.selectbox("Qual região você mora?", ['southwest', 'northwest', 'northeast', 'southeast'])
info_seg = st.number_input("Digite o valor do seu seguro saúde")

# Dicionários para mapear as entradas do usuário
map_fumante = {"Sim": 1, "Não": 0}
map_regiao = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
map_sexo = {'Masculino': 1, 'Feminino': 0, 'Prefiro não dizer': -1}  # Supondo que não é usado na previsão, ou pode-se descartar

# Prever
if st.button("PREVER"):
    df_previsao = pd.DataFrame({
        'idade': [info_idade],
        'sexo': [map_sexo[info_sexo]],
        'seguro saúde': [info_seg],
        'crianças': [total_filho],
        'fumante': [map_fumante[info_fumante]],
        'região': [map_regiao[info_regiao]]
    })
    df_previsao_scaled = scaler.transform(df_previsao)
    # Fazer a previsão
    predicao = modelo.predict(df_previsao_scaled)
    st.write(f"O valor previsto para o seguro é: {predicao[0]:.2f} reais")
