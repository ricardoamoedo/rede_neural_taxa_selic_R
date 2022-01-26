# Previsão da taxa selic utilizando redes neurais em R
# arquivo disponível em https://www.kaggle.com/felsal/ibovespa-stocks/version/26

# ativando pacotes (bibliotecas)
library(neuralnet)
library(quantmod)
library(grt)
library(zoo)
library(forecast)

# importanto os dois dataframes
df = read.csv("b3_stocks_1994_2020.csv")
df2 = read.csv("selic.csv")

# mudando o tipo de dados da coluna 1 para data
df$datetime = as.Date(df$datetime)
df2$datetime = as.Date(df2$datetime)

# filtrando dados a partir de 01/01/2018
df = subset(df, df$datetime > "2017-01-01")
df2 = subset(df2, df2$datetime > "2017-01-01")

df2["selicm1"] = Lag(df2$selic,1)
df2["selicm2"] = Lag(df2$selic,2)
df2["selicm3"] = Lag(df2$selic,3)
df2["selicm4"] = Lag(df2$selic,4)
df2["selicm5"] = Lag(df2$selic,5)

# Retirando valores faltantes
df2 = na.omit(df2)
df3 = df2
df3$datetime = NULL

# Normalizando os dados para colocar na rede neural 
df2_scale = scale(df3)


# Aplicando os dados na rede neural (neuralnet)
# com 2 camadas ocultas, a primeira com dois neurônios e a segunda com 3 neurônios
nn = neuralnet(selic ~ selicm1 + selicm2 + selicm3 + selicm4 + selicm5, 
               data = df2_scale, hidden = c(2,3), threshold = 1, stepmax = 1000)

plot(nn)

# criando variável preditora
previsao = predict(nn, df2_scale)

# desfazeno a normalização
previsao = unscale(previsao)

# Criando tabela com colunas data e dados da previsão
prev = as.character(df2$datetime)
prev = cbind(prev, previsao)

# plotando o gráfico da selic e das previsões
plot(as.Date(df2$datetime), as.vector(df2$selic), col = 'blue', type = 'l')
plot(as.Date(prev[,1]), as.vector(prev[,2]), col = 'red', type = 'l')

# verificando a acurácia das previsões
accuracy(as.vector(previsao), df2$selic)



