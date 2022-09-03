
"""# 1. Análise descritiva dos dados
Após copiar o arquivo, vamos carregar as bibliotecas base para fazer a análise descritiva
"""

import streamlit as st
import pandas as pd
import numpy as np 
import seaborn as sns
import regex as re
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import random
import os
import matplotlib.patheffects as path_effects
import itertools
from scipy import stats
import math

def descritiva_tabela(data_frame_selec):
    df2 = data_frame_selec.describe(include = 'all')
    df2.loc['dtype'] = data_frame_selec.dtypes
    df2.loc['size'] = len(data_frame_selec)
    df2.loc['% count'] = data_frame_selec.isnull().mean()
    return pd.DataFrame(df2.transpose())

def add_median_labels(ax, fmt='.1f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

"""Verificando missings, quantidades e quantis"""

#os.chdir('D:\\Users\\krums\\Documents\\MBA FIAP IA\Bootcamp IA & Machine Learning')
df_creditos = pd.read_csv('Database\\solicitacoescredito.csv')

df_creditos.head()

descritiva_tabela(df_creditos)

"""### 1.1 Analisando e filtrando de casos inconsistentes
Para a amostra, temos que verificar:
1. Casos de clientes que tem mais de um nome de `razaoSocial`
2. Casos com solicitação de valor zero
"""

df_creditos.loc[df_creditos['valorSolicitado']<=0.0]

df_creditos_count_razao = df_creditos.groupby('cnpjSemTraco')['razaoSocial'].nunique().reset_index().rename(columns={'razaoSocial':'qtde_razaoSocial'})

#8 solicitações provenientes de clientes com mais de uma razão social
df_creditos.merge(df_creditos_count_razao[df_creditos_count_razao['qtde_razaoSocial']>1],on=['cnpjSemTraco'],how='inner').head()

df_creditos = df_creditos.merge(df_creditos_count_razao[df_creditos_count_razao['qtde_razaoSocial']==1],on=['cnpjSemTraco'],how='inner')

"""### 1.2 Criando o dataset para clientes

No caso temos que pegar as features dos clientes que resumem cada um deles. Podemos pegar as features em função to tempo (agregando variaveis por solicitação para cada cliente), porém testaremos a clusterização sem usar a variação temporal ainda. Se isso já ajudar a descrever é um bom progresso:
"""

#features calculadas (em função do tempo)
#buscando a linha índice de solicitação

df_clientes_aux = df_creditos.groupby(df_creditos['cnpjSemTraco'])['numero_solicitacao'].min().reset_index().rename(columns={'numero_solicitacao':'min_solicitacao'})

df_clientes = df_creditos.merge(df_clientes_aux,left_on=['cnpjSemTraco','numero_solicitacao'], right_on=['cnpjSemTraco','min_solicitacao'],how='inner').copy()

"""Devemos selecionar as variáveis que maximizam o tamanho da amostra. para isso, olhamos todas as variáveis pertinentes aos clientes e verificamos quanto de missing temos. O conjunto de colunas deve ter o menor % de missings possivel."""

df1 = descritiva_tabela(df_clientes)
df1.filter(['cnpjSemTraco','maiorAtraso'
            ,'margemBrutaAcumulada','percentualProtestos','primeiraCompra'
            ,'faturamentoBruto','margemBruta','periodoDemonstrativoEmMeses','custos'
            ,'anoFundacao','capitalSocial','restricoes','empresa_MeEppMei'
            ,'scorePontualidade'],axis=0)['% count'].max()

df_clientes = df_clientes.filter(['cnpjSemTraco','maiorAtraso','valorSolicitado'
                                 ,'margemBrutaAcumulada','percentualProtestos','primeiraCompra'
                                 ,'faturamentoBruto','margemBruta','periodoDemonstrativoEmMeses','custos'
                                 ,'anoFundacao','capitalSocial','restricoes','empresa_MeEppMei'
                                 ,'scorePontualidade'],axis=1)

"""Agora temos que fazer um hot encode de algumas variáveis:

1. tirar todos os nulos de variáveis de cliente;
2. Podemos usar como default da `primeiraCompra` pra quem não tem como sendo uma data 1900-01-01;
3. `restricoes` deve ser tratada com hot encode;
"""

df_desc_cli = descritiva_tabela(df_clientes)
df_desc_cli

#1.
df_clientes['primeiraCompra'].fillna('1900-01-01 00:00:00', inplace=True)
df_clientes['primeiraCompra'] = df_clientes['primeiraCompra'].apply(lambda x:re.sub('T',' ',x))
#Garantindo que teremos valores de data decentes:
df_clientes.loc[df_clientes['primeiraCompra']<='1900-01-01 00:00:00',['primeiraCompra']]='1900-01-01 00:00:00'
df_clientes['primeiraCompra'] = pd.to_datetime(df_clientes['primeiraCompra'], format='%Y-%m-%d %H:%M:%S')

#2. mudando as colunas categorias em hot encoding (para as dicotomicas é mais fácil)
df_clientes['restricoes'] = df_clientes['restricoes']*1
df_clientes['restricoes'] = df_clientes['restricoes'].apply(float)
df_clientes['empresa_MeEppMei'] = df_clientes['empresa_MeEppMei']*1
df_clientes['empresa_MeEppMei'] = df_clientes['empresa_MeEppMei'].apply(float)
df_clientes['maiorAtraso'] = df_clientes['maiorAtraso'].apply(float)

#3. 
df_clientes.dropna(inplace=True)
df_clientes.reset_index(inplace=True)

#a clusterização deve ter variáveis em função das características do cliente, ou seja:
X = df_clientes.filter(['margemBrutaAcumulada','percentualProtestos','empresa_MeEppMei'
                        ,'faturamentoBruto','margemBruta','custos','margemBruta_neg','custos_neg'
                        ,'anoFundacao','capitalSocial','restricoes','scorePontualidade']
,axis=1).copy()

"""Padronizando as variáveis escolhidas para terem o mesmo peso:"""

X_scale=pd.DataFrame(scale(X),columns=X.columns.values)
random.seed(1994)
pca = PCA()
X_reduced = pca.fit_transform(X_scale)
df_comp = pd.DataFrame(X_reduced,columns=['PC'+str(num_c) for num_c in range(0,pca.n_components_)])
df_comp.head()

"""Verificando o % de explicação por componente principal"""

np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

kmeans = KMeans(4, random_state=0)
labels_KMEANS = kmeans.fit(df_comp).predict(df_comp)

gmm = GaussianMixture(n_components=4).fit(df_comp)
labels_GMM = gmm.predict(df_comp)

df_comp['labels_KMEANS']=labels_KMEANS
df_comp['labels_GMM']=labels_GMM
X['labels_KMEANS']=labels_KMEANS
X['labels_GMM']=labels_GMM

"""Visualizando com componentes principais"""

from turtle import title

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))

ax1.scatter(df_comp['PC0'], df_comp['PC1'], c=df_comp['labels_KMEANS'], s=40, cmap='viridis')
ax1.set_title('Clusterização por K-Means')
ax2.scatter(df_comp['PC0'], df_comp['PC1'], c=df_comp['labels_GMM'], s=40, cmap='viridis')
ax2.set_title('Clusterizção por GMM')

for ax in fig.axes:
    ax.set_xlabel('PC0')
    ax.set_ylabel('PC1')

"""Visualizando por cada variável

Primeiramente por KMEANS
"""

#sns.pairplot(X, hue="labels_KMEANS")

"""Por GMM"""

#sns.pairplot(X, hue="labels_GMM")

"""Verificando métricas entre grupos, e também a variancia entre grupos:"""

from scipy.spatial import distance
matrix_dist = distance.cdist(X, X, 'euclidean')

pd_distancias = pd.DataFrame(matrix_dist,columns=X.index.values)
pd_distancias = pd_distancias.stack()
pd_distancias.name = 'distancia'
pd_distancias = pd_distancias.reset_index()

X_labels = X[['labels_KMEANS','labels_GMM']]
X_labels['index_chave'] = X_labels.index.values
X_labels_1 = X_labels.rename({'labels_KMEANS':'labels_KMEANS_1','labels_GMM':'labels_GMM_1','index_chave':'index_chave'},axis='columns').copy()
X_labels_2 = X_labels.rename({'labels_KMEANS':'labels_KMEANS_2','labels_GMM':'labels_GMM_2','index_chave':'index_chave'},axis='columns').copy()
pd_distancias = pd_distancias.merge(X_labels_1,left_on='level_0',right_on='index_chave',how='inner')
pd_distancias = pd_distancias.merge(X_labels_2,left_on='level_1',right_on='index_chave',how='inner')
dist_entre_GMM = pd_distancias[pd_distancias['labels_KMEANS_2']!=pd_distancias['labels_KMEANS_1']]['distancia'].sum()
dist_entre_KMEANS=pd_distancias[pd_distancias['labels_GMM_2']!=pd_distancias['labels_GMM_1']]['distancia'].sum()

razao_vars = dist_entre_KMEANS/dist_entre_GMM
if razao_vars>1:
    resp = 'maior'
else:
    resp='menor'

"""Separar na descritiva de categóricas e numericas, tendo discretas e continuas:"""

df_clientes['index'] = df_clientes.index
df_clientes = df_clientes.merge(X_labels,left_on='index',right_on='index_chave',how='inner').copy()
df_clientes.labels_KMEANS = df_clientes.labels_KMEANS.astype("category")
df_clientes.labels_GMM = df_clientes.labels_GMM.astype("category")
df_clientes.drop(['index_chave'],axis=1,inplace=True)

colunas_cont = ['margemBrutaAcumulada','percentualProtestos'
                        ,'faturamentoBruto','margemBruta','custos'
                        ,'anoFundacao','capitalSocial','scorePontualidade']

labels_cols=['labels_GMM','labels_KMEANS','index']

# empilhando as variáveis num formato longo:
# all_columns = df_clientes.columns
df_clientes_emp  = df_clientes.melt(id_vars=labels_cols)
df_clientes_emp_c = df_clientes_emp.loc[[elm in colunas_cont for elm in df_clientes_emp['variable'].values]]
# df_clientes_emp_c['variable'].drop_duplicates()

"""Trazendo em formato de tabela"""

#lembrando da distribuição de empresas por cluster:
df_barplot = df_clientes[['labels_GMM','labels_KMEANS']].melt()

g = sns.catplot(x="variable", hue='value',kind='count', data=df_barplot,aspect=2)
ax = g.axes[0][0]

for p in ax.patches:
    ax.annotate('{:,d}'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
        textcoords='offset points')

"""Como pode-se perceber, a distribuição da clusterização com GMM é mais homoegena, e por isso pode ser um motivo para ser mais recomendável"""

print('A distancia entre grupos de KMEANS é {razao_vars}% {resp} do que o agrupamento por GMM'.format(razao_vars=round(abs(1-razao_vars)*100,2), resp=resp))

"""POr mais que a clusterização por KMEANS tenha um poder de distinguir os grupos 12% maior, sua distribuição não é homogenea.

Assim sendo, seguimos pela clusterização com GMM
"""

#colunas para resumo:
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format','{:,.2f}'.format)
labels_selec='labels_GMM'
cols_selec_resumo = [elm for elm in df_clientes.columns if elm not in ['index','cnpjSemTraco','labels_KMEANS','primeiraCompra','periodoDemonstrativoEmMeses']]
describe_clus = df_clientes.filter(cols_selec_resumo,axis=1).groupby(labels_selec).describe(include='all',percentiles = [0.1,0.25,0.5,0.75,0.9],datetime_is_numeric=True).transpose()
# inferno na terra para selecionar essa porcaria:
#criando o index que possui todas as variáveis que queremos:
lista_indices = list(itertools.product(cols_selec_resumo,['min','10%','25%','50%','75%','90%','max']))
describe_clus.filter(lista_indices,axis=0)
# posso criar graficos de barra para cada valor desse describe por grupo, sendo minimo, media e maximo como barras diferentes lado a lado por grupo, por variável

from turtle import width

sns.set_theme(style="ticks") 

# Initialize the figure with a logarithmic x axis

# df_describe_vars = pd.DataFrame(list(describe_clus.index.values),columns=['var_cliente','var_describe'])
df_describe = describe_clus.filter(lista_indices,axis=0).reset_index()
df_describe.rename({'level_0':'var_cliente','level_1':'var_describe'},axis='columns',inplace=True)
df_describe = df_describe.melt(id_vars=['var_describe','var_cliente'])
df_describe_1 = df_describe.loc[[vard not in ['margemBruta','margemBrutaAcumulada','faturamentoBruto','custos','capitalSocial'] for vard in df_describe['var_cliente']]]
df_describe_2 = df_describe.loc[[vard in ['margemBruta','margemBrutaAcumulada','faturamentoBruto','custos','capitalSocial'] for vard in df_describe['var_cliente']]]
# df_describe.drop(['labels_KMEANS'],axis=1)

# sea = sns.FacetGrid(df_describe, row = "var_cliente")
#sea.map(sns.barplot, "var_describe", "pulse",order = ["no fat", "low fat"])

#g = sns.catplot(data=df_describe_1, x="var_describe", y="value",col="var_cliente",hue='labels_GMM'
#                ,kind='bar',sharey=False,aspect=4,height=5,col_wrap=2)

for ax in g.axes:
    for p in ax.patches:
             ax.annotate('{:,.2f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                 textcoords='offset points')

"""Gráficos com escala logarítmica (para melhor vosualização)"""

#g = sns.catplot(data=df_describe_2, x="var_describe", y="value",col="var_cliente",hue='labels_GMM'
#                ,kind='bar',sharey=False,aspect=3,height=5,col_wrap=2)

for ax in g.axes:
    for p in ax.patches:
             ax.annotate('{:,.2f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                 textcoords='offset points')
             ax.set(yscale="log")

"""Com estes graficos de barras resumindo percentis de cada variável quantitativa dos clusters, podemos assumir algumas premissas sobre os grupos. Abaixo enumeramos caracteristicas inerentes a cada um deles para podermos depois determinarmos a segmentação para futuros clientes:

1. **Tier Bronze - Meia idade**: Este tier contempla empresas que possuem um perfil de margem bruta declarada absoluta menor que todas as outras no geral. Pelo menos 50% desse conjunto não possui margem bruta declarada. Outra variável que foi possível notar é de atraso, que não supera 33 dias em 90% dos casos, mas que podem extrapolar para quase 3 anos. Sua margem percentual está no perfil mediano de toda a base. Estas empresas não possuem restrições. Estas também não possuem custos declarados para pelo menos 75% dos casos.
2. **Tier Mei - Novas em folha**: Este tier possui empresas com a margem bruta um pouco maior que o **Tier Bronze**, tendo porém o acumulado % menor que este. Neste conjunto tiveram atrasos também significativos, em que pelo menos 10% da base atrasou 46 dias ou mais. Toda a base se enquadra em micro e pequenas empresas (Mei), e com isso, possuem o menor faturamento na sua grande maioria quando comparado com os demais clusters. Com isso, estas são empresas que estão em fase de inicio do crescimento.
3. **Tier Ouro - Antigas**: As empresas neste grupo são de um perfil mais rentável. Analisando a distribuição de anos de fundação, elas são as mais antigas quando comparando com os outros clusteres. Possuem um grau de atraso mais frequente em todo o conjunto em comparação com as demais. Existem 25% das empresas nesse grupo com algum tipo de restrição. Sendo assim, por mais que existem restrições e atrasos, isso é balanceado pela robustez de seu faturamento, que é bem maior que as demais, passando em um grau de ordem 10 vezes maior na maioria dos casos.
4. **Tier Prata - Risco Alto**: Esse conjunto contemplam empresas que aparentam estar em fase de cresimento progressivo, tendo uma margem % e bruta menor que todas as outras. Além disso, estas empresas possuem restrições em pelo menos 75% dos casos. Estas também possuem o menor perfil geral de captalização, quando comparadas com as demais. Neste tier existe um potencial de ganho, mas existem varios alertas a se considerar quanto a solidez da empresa.

Dadas essas caracterizações, podemos criar um modelo de classificação que utiliza-se destas para determinar novos casos. Para isso vamos usar um algorítmo de classificação robusto:
"""

X

X_train, X_test, y_train, y_test = train_test_split(X, X.labels_GMM, random_state=2,train_size=0.7)
rbf_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='rbf', C=5))
])
rbf_kernel_svm_clf.fit(X_train, y_train)

"""Testando para alguns casos da internet para ver em que cenário elas caem"""

from sklearn.metrics import plot_confusion_matrix
y_pred = rbf_kernel_svm_clf.predict(X_test)
plot_confusion_matrix(rbf_kernel_svm_clf, X_test, y_test)  
plt.show()

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
print('A precisão ficou em'
,precision_score(y_test, y_pred,average='weighted'),'e o recall ficou em'
,recall_score(y_test, y_pred,average='weighted'),'sendo que no geral a acurácia é',accuracy_score(y_test, y_pred))

"""# 3. Modelagem para análise de crédito
Para o modelo de análise de credito, podemos dividir em:

- Modelo para primeira compra
- Modelo para clientes recorrentes

Carregando bibliotecas necessárias
"""

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

#Função para tirar NaN e Outliers (1%)
def ajusta_outliers(df, x):

    lower_limit = df[x].quantile(0.01)  
    upper_limit = df[x].quantile(0.99)

    df[x].fillna(-9999999, inplace=True)

    df[x] = np.where((df[x] < lower_limit) & (df[x] != -9999999), lower_limit, df[x])
    df[x] = np.where((df[x] > upper_limit) & (df[x] != -9999999), upper_limit, df[x])
    return df

df = df_creditos.drop(['numero_solicitacao', 'razaoSocial','nomeFantasia','cnpjSemTraco','status','definicaoRisco','diferencaPercentualRisco','dashboardCorrelacao','intervaloFundacao'], axis=1)

"""### 3.1 Pré processamento de base
Abaixo estão algumas tratativas de base
"""

#Ajustar datas
df['dataConsiderada'] = df['dataAprovadoNivelAnalista'].where(df['dataAprovadoNivelAnalista'] > '0', df['dataAprovadoEmComite'])
df['dataConsiderada'] = pd.to_datetime(df['dataConsiderada'], errors='coerce')
df['primeiraCompra'] = pd.to_datetime(df['primeiraCompra'], errors='coerce')
df['periodoBalanco'] = pd.to_datetime(df['periodoBalanco'], errors='coerce')
df = df.drop(['dataAprovadoNivelAnalista','dataAprovadoEmComite'], axis=1)

#Calcular diferença de datas
df['idadeNaSolicitacao'] = pd.DatetimeIndex(df['dataConsiderada']).year - df['anoFundacao']
df['periodoBalanco_Anos'] = pd.DatetimeIndex(df['dataConsiderada']).year - pd.DatetimeIndex(df['periodoBalanco']).year
df['primeiraCompra_Anos'] = pd.DatetimeIndex(df['dataConsiderada']).year - pd.DatetimeIndex(df['primeiraCompra']).year
df = df.drop(['anoFundacao','periodoBalanco','primeiraCompra','dataConsiderada'], axis=1)

#Criar dummys de EPP e Restrições
df = pd.get_dummies(df)
df = df.drop(['empresa_MeEppMei_False','restricoes_False'], axis=1)

#Remove Outliers e NaN
df = ajusta_outliers(df,'percentualProtestos')
df = ajusta_outliers(df,'ativoCirculante')
df = ajusta_outliers(df,'passivoCirculante')
df = ajusta_outliers(df,'totalAtivo')
df = ajusta_outliers(df,'totalPatrimonioLiquido')
df = ajusta_outliers(df,'endividamento')
df = ajusta_outliers(df,'duplicatasAReceber')
df = ajusta_outliers(df,'estoque')
df = ajusta_outliers(df,'faturamentoBruto')
df = ajusta_outliers(df,'margemBruta')
df = ajusta_outliers(df,'custos')
df = ajusta_outliers(df,'capitalSocial')
df = ajusta_outliers(df,'scorePontualidade')
df = ajusta_outliers(df,'limiteEmpresaAnaliseCredito')
df = ajusta_outliers(df,'idadeNaSolicitacao')
df = ajusta_outliers(df,'periodoBalanco_Anos')
df = ajusta_outliers(df,'primeiraCompra_Anos')
df = ajusta_outliers(df,'valorAprovado')

df['Aprovado_True'] = np.where((df['valorAprovado'] < 0) , 0, 1)

"""Analisando como as colunas ficaram após o tratamento"""

descritiva_tabela(df)

"""Preparando a base com veriaveis e respotas"""

# tirando uma cópia para preservar o DataFrame original
df_copia = df.copy()

# definindo os recuros de entrada(X) e saida(y) 
X = df_copia.drop(['valorAprovado','Aprovado_True'], axis=1)
X = X.iloc[: , :].values
y = df_copia['Aprovado_True'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.30, 
                                                    shuffle=True, 
                                                    random_state = 0)

# verificando a dimensão dos dados de treino e teste
print(f'Dados para Treino: {X_train.shape[0]} amostras')
print(f'Dados para Teste: {X_test.shape[0]} amostras')

"""### 3.2 Preparando o modelo para treino
#### 3.2.1 Modelo de concessão de credito (sim ou não) 
"""

from xgboost import XGBClassifier
clf = XGBClassifier(random_state=0)

# Training the XGB classifier
clf.fit(X_train, y_train)

"""Fazendo a predição com os dados de teste"""

y_pred = clf.predict(X_test)

"""Para verificar a performance do modelo, visualizamos os resultados pela Matriz de Confusão"""

from sklearn.metrics import plot_confusion_matrix
#plot_confusion_matrix(clf, X_test, y_test, cmap='Blues')

"""Score da Acurácia"""

print(f'Precisão do modelo : {round(accuracy_score(y_test, y_pred)*100,3)}%')

new_df = df.copy()
new_df = new_df.loc[df['Aprovado_True'] == 1]
new_df = new_df.drop(['Aprovado_True'], axis=1)

"""##### 3.2.2 Modelo de volume de concessão de crédito
Antes de fazer o treino, é necessário ainda fazer pré processamento da coluna do valor aprovado para determinar um range de buckets para previsão
"""

labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
new_df['Bin'] = pd.qcut(new_df['valorAprovado'], 20, labels=labels,duplicates='drop') #
new_df = new_df.drop(['valorAprovado'], axis=1)

"""Definindo os recuros de entrada(X) e saida(y) """

X2 = new_df.drop(['Bin'], axis=1)
X2 = X2.iloc[: , :].values
y2 = new_df['Bin'].values

# separando os dados de treino e teste
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, 
                                                    test_size = 0.30, 
                                                    shuffle=True, 
                                                    random_state = 0)

# verificando a dimensão dos dados de treino e teste
print(f'Dados para Treino: {X_train2.shape[0]} amostras')
print(f'Dados para Teste: {X_test2.shape[0]} amostras')

"""Treinando o modelo"""

#Preparando o modelo e executando
clf2 = XGBClassifier(random_state=0)

# Training the XGB classifier
clf2.fit(X_train2, y_train2)

"""Fazendo a predição com os dados de teste"""

y_pred2 = clf2.predict(X_test2)

"""Apresentando os resultados pela Matriz de Confusão"""

from sklearn.metrics import plot_confusion_matrix
fig, ax = plt.subplots(figsize=(10, 10))
plot_confusion_matrix(clf2, X_test2, y_test2, cmap='Blues',ax=ax)