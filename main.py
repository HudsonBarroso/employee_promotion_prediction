# Importação de pacotes
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# Criando um título para nosso projeto
st.title("Avaliando se você será promovido ou não")
st.write('''
### Preencha os campos para realizar a previsão!
''')

st.sidebar.header('Dados do Funcionário')

department_mapping = {
                'Sales & Marketing':1,
                'Operations':2,
                'Technology':3,
                'Analytics':4,
                'R&D':5,
                'Procurement':6,
                'Finance':7,
                'HR':8,
                'Legal':9
}

education_mapping = {
                'Mestrado/Doutorado/PosDoc':1,
                'Superior Completo':2,
                'Ensino Médio/Fundamental':3
}

kpi_mapping = {
                'Sim':1,
                'Não':0
}

award_mapping = {
                'Sim':1,
                'Não':0
}

# Criando o menu lateral para inserir os dados do funcionário
def user_input_features():
    department_feature = st.sidebar.selectbox("Selecione o Departamento", ("Sales & Marketing", "Operations",
                                                                           "Technology", "Analytics", "R&D",
                                                                           "Procurement", "Finance", "HR",
                                                                           "Legal"))
    department = department_mapping[department_feature]
    education_feature = st.sidebar.selectbox("Selecione a sua Escolaridade", ("Mestrado/Doutorado/PosDoc",
                                                                              "Superior Completo",
                                                                              "Ensino Médio/Fundamental"))
    education = education_mapping[education_feature]
    kpi_met = st.sidebar.selectbox("Alcançou as metas de pelos menos 80%", ("Sim", "Não"))
    kpi = kpi_mapping[kpi_met]
    award_won = st.sidebar.selectbox("Foi Premiado este ano", ("Sim", "Não"))
    award = award_mapping[award_won]
    age = st.sidebar.slider("Selecione a sua idade?", 15, 70, 30)
    no_of_training = st.sidebar.slider("Quantos treinamentos você realizou?", 1, 10, 2)
    avg_training_score = st.sidebar.slider("Média das notas no treinamento", 40, 99, 70)
    previous_year_rating = st.sidebar.slider("Avaliação de Performance ano anterior", 1, 5, 4)
    length_of_service = st.sidebar.slider("Tempo de serviço", 1, 13, 5)

    # train['awards_won?']+train['KPIs_met >80%'] + train['previous_year_rating']
    sum_metric = award + kpi + previous_year_rating

    # train['total_score'] = train['avg_training_score'] * train['no_of_trainings']
    total_score = avg_training_score * no_of_training

    data = {
        'department': department,
        'education': education,
        'no_of_trainings': no_of_training,
        'age': age,
        'previous_year_rating': previous_year_rating,
        'length_of_service': length_of_service,
        'KPIs_met >80%': kpi,
        'awards_won?': award,
        'avg_training_score': avg_training_score,
        'sum_metric': sum_metric,
        'total_score': total_score,
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# lendo o dataset de teste
promotion_test = pd.read_csv('./input/test.csv')
# concatenando os dados do usuário com os dados do dataset de teste
df = pd.concat([input_df, promotion_test], axis=0)

# selecionando a primeira linha (o valor inserido pelo usuário)
df = df[:1]

# realizando a leitura do modelo salvo
load_randomForest = pickle.load(open('employee_promotion_prediction.pkl', 'rb'))

# aplicando o modelo para realizar a previsão
prediction = load_randomForest.predict(input_df)
prediction_probability = load_randomForest.predict_proba(input_df)

st.subheader('Previsão')
result = np.array(['Você provavelmente não será promovido.','Você será promovido!'])
st.write(result[prediction][0])

st.subheader('Probabilidade da Previsão')
st.write('Baseado nos dados selecionados,\nvocê tem {0:.2f}% de chances de ser promovido.'.format(
    prediction_probability[0][1] * 100))

if prediction == 0:
    st.image('./images/not_promoted.jpg', use_column_width=True)
else:
    st.image('./images/promoted.jpg', use_column_width=True)