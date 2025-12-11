import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st
import streamlit.components.v1 as components

sns.set_theme(context='talk', style='ticks')

st.set_page_config(
    page_title='Projeto 2 - Análise de Renda',
                   page_icon=':bar_chart:',
                   layout='wide')

st.title('Projeto 2 - Análise de Renda')

renda = pd.read_csv('C:/Users/anado/OneDrive/Documentos/EBAC/M16 - Métodos de análise/dados.csv')
renda.head()

prof = ProfileReport(renda, explorative=True, minimal=True)
prof.to_file('C:/Users/anado/OneDrive/Documentos/EBAC/M16 - Métodos de análise/analise_renda.html')

st.write('## Análise Exploratória dos Dados')

st.write('### Estatísticas Descritivas')
st.write(renda.describe())

st.markdown('----------')

st.write('### Análise Univariada')
HtmlFile = open("C:/Users/anado/OneDrive/Documentos/EBAC/M16 - Métodos de análise/analise_renda.html", 'r', encoding='utf-8')
source_code = HtmlFile.read()
components.html(source_code, height=1000, scrolling=True)

st.markdown('----------')

st.write('### Análise Bivariada')
st.write('##### Gráficos de Distribuição')

col1, col2 = st.columns(2)

with col1:
    fig1, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10), sharey=True)

    sns.pointplot(x='sexo', y='mau', data=renda, dodge=True, errorbar=('ci', 90), ax=axes[0, 0])
    axes[0, 0].set_title('Taxa de Mau Pagador por Gênero')

    sns.pointplot(x='posse_de_veiculo', y='mau', data=renda, dodge=True, errorbar=('ci', 90), ax=axes[0, 1])
    axes[0, 1].set_title('Taxa de Mau Pagador por Posse de Veículo')

    sns.pointplot(x='posse_de_imovel', y='mau', data=renda, dodge=True, errorbar=('ci', 90), ax=axes[0, 2])
    axes[0, 2].set_title('Taxa de Mau Pagador por Posse de Imóvel')

    sns.pointplot(x='possui_fone_comercial', y='mau', data=renda, dodge=True, errorbar=('ci', 90), ax=axes[1, 0])
    axes[1, 0].set_title('Taxa de Mau Pagador por Posse de Fone Comercial')

    sns.pointplot(x='possui_fone', y='mau', data=renda, dodge=True, errorbar=('ci', 90), ax=axes[1, 1])
    axes[1, 1].set_title('Taxa de Mau Pagador por Posse de Fone')

    sns.pointplot(x='possui_email', y='mau', data=renda, dodge=True, errorbar=('ci', 90), ax=axes[1, 2])
    axes[1, 2].set_title('Taxa de Mau Pagador por Posse de E-mail')

    plt.tight_layout()

    # show in streamlit
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    fig2, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10), sharey=True)

    sns.pointplot(x='qtd_filhos', y='mau', data=renda, dodge=True, errorbar=('ci', 90), ax=axes[0, 0])
    axes[0, 0].set_title('Taxa de Mau Pagador por Quantidade de Filhos')

    sns.pointplot(x='tipo_renda', y='mau', data=renda, dodge=True, errorbar=('ci', 90), ax=axes[0, 1])
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
    axes[0, 1].set_title('Taxa de Mau Pagador por Tipo de Renda')

    sns.pointplot(x='educacao', y='mau', data=renda, dodge=True, errorbar=('ci', 90), ax=axes[0, 2])
    axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=30)
    axes[0, 2].set_title('Taxa de Mau Pagador por Nível de Educação')

    sns.pointplot(x='estado_civil', y='mau', data=renda, dodge=True, errorbar=('ci', 90), ax=axes[1, 0])
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=30)
    axes[1, 0].set_title('Taxa de Mau Pagador por Estado Civil')

    sns.pointplot(x='tipo_residencia', y='mau', data=renda, dodge=True, errorbar=('ci', 90), ax=axes[1, 1])
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
    axes[1, 1].set_title('Taxa de Mau Pagador por Tipo de Residência')

    sns.pointplot(x='qt_pessoas_residencia', y='mau', data=renda, dodge=True, errorbar=('ci', 90), ax=axes[1, 2])
    axes[1, 2].set_title('Taxa de Mau Pagador por Qt Pessoas na Residência')

    plt.tight_layout()

    # show in streamlit
    st.pyplot(fig2)
    plt.close(fig2)

col1, col2 = st.columns(2)

with col1:
    renda['idade_bin'] = pd.cut(renda['idade'], bins=[18, 25, 35, 45, 55, 65, 100])

    plt.figure(figsize=(10,4))
    sns.barplot(x='idade_bin', y='mau', data=renda, errorbar=('ci', 95))
    plt.title('Taxa de Mau Pagador por Faixa de Idade')

    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

with col2:
    renda['tempo_emprego_bin'] = pd.cut(renda['tempo_emprego'], bins=[0, 1, 3, 5, 10, 20, 50])

    plt.figure(figsize=(10,4))
    sns.barplot(x='tempo_emprego_bin', y='mau', data=renda, errorbar=('ci', 95))
    plt.title('Taxa de Mau Pagador por Tempo de Emprego')

    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

plt.figure(figsize=(14,10))
corr = renda.corr(numeric_only=True)
corr_mau = corr['mau'].sort_values(ascending=False)
top10 = corr_mau.abs().sort_values(ascending=False).head(11).index  # inclui 'mau'

st.write('##### Correlações entre Variáveis Numéricas')

col1, col2 = st.columns(2)

with col1:
    st.write(corr_mau)

with col2:
    plt.figure(figsize=(10, 8))
    sns.heatmap(renda[top10].corr(numeric_only=True), annot=True, fmt=".2f", cmap='rocket')
    plt.title('Top 10 correlações com a variável mau')

    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()
    
st.markdown('----------')

st.write('## Modelagem Preditiva')

metadata = pd.DataFrame(renda.dtypes, columns=['tipo'])
metadata['n_categorias'] = [renda[var].nunique() for var in metadata.index]

def convert_dummy(df, feature, rank=0):
    df = df.copy()  # segurança
    dummies = pd.get_dummies(df[feature], prefix=feature)

    # categoria mais frequente → será dropada (evita multicolinearidade)
    mode = df[feature].value_counts().index[rank]
    col_to_drop = f"{feature}_{mode}"

    dummies = dummies.drop(col_to_drop, axis=1)
    df = df.drop(feature, axis=1)

    return df.join(dummies)

df = renda.copy()

for var in metadata[metadata['tipo'] == 'object'].index:
    df = convert_dummy(df, var)

# separando features e target
X = df.drop("mau", axis=1)
y = df["mau"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight='balanced'
)

# converter para string (se já não for)
X_train['idade_bin'] = X_train['idade_bin'].astype(str)
X_train['tempo_emprego_bin'] = X_train['tempo_emprego_bin'].astype(str)
X_test['idade_bin']  = X_test['idade_bin'].astype(str)
X_test['tempo_emprego_bin']  = X_test['tempo_emprego_bin'].astype(str)

# one-hot (drop_first opcional)
X_train_ohe = pd.get_dummies(X_train, drop_first=True)
X_test_ohe  = pd.get_dummies(X_test, drop_first=True)

# alinhar colunas (técnica importante)
X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)

# atualizar referências
X_train = X_train_ohe
X_test  = X_test_ohe

# treinar
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

st.write('### Resultados do Modelo de Classificação')

st.write(f"##### Acurácia: {accuracy_score(y_test, y_pred):.4f}")

st.write('#### Matriz de Confusão')
st.write(confusion_matrix(y_test, y_pred))

st.write('#### Relatório de Classificação')
st.text(classification_report(y_test, y_pred))