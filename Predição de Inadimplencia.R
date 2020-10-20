# Predição de Inadimplência de Clientes

# Instalando os pacotes para o projeto (os pacotes precisam ser instalados apenas uma vez)
install.packages("Amelia")
install.packages("caret")
install.packages("lattice")
install.packages("lava")
install.packages("purrr")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape2")
install.packages("randomForest")
install.packages("e1071")

# Carregando os pacotes 
library(Amelia)
library(ggplot2)
library(caret)
library(reshape2)
library(randomForest)
library(dplyr)
library(e1071)
library(purrr)
library(lattice)
library(lava)

# Carregando os datasets
dataset <- read.csv("credit-card.csv")

# Visualizando os dados e sua estrutura
View(dataset)
str(dataset) 
head(dataset) 

##Transformando e Limpando os Dados ##

# Convertendo os atributos idade, sexo, escolaridade e estado civil para fatores (categorias)

# Idade
head(dataset$AGE) 
dataset$AGE <- cut(dataset$AGE, c(0,30,50,100), labels = c("Jovem","Adulto","Idoso"))
head(dataset$AGE) 

# Sexo
dataset$SEX <- cut(dataset$SEX, c(0,1,2), labels = c("Masculino","Feminino"))
head(dataset$SEX) 

# Escolaridade
dataset$EDUCATION <- cut(dataset$EDUCATION, c(0,1,2,3,4), 
                         labels = c("Pos Graduado","Graduado","Ensino Medio","Outros"))
head(dataset$EDUCATION) 

# Estado Civil
dataset$MARRIAGE <- cut(dataset$MARRIAGE, c(-1,0,1,2,3),
                        labels = c("Desconhecido","Casado","Solteiro","Outros"))
head(dataset$MARRIAGE) 

# Convertendo a variavel que indica pagamentos para o tipo fator
dataset$PAY_0 <-as.factor(dataset$PAY_0)
dataset$PAY_2 <-as.factor(dataset$PAY_2)
dataset$PAY_3 <-as.factor(dataset$PAY_3)
dataset$PAY_4 <-as.factor(dataset$PAY_4)
dataset$PAY_5 <-as.factor(dataset$PAY_5)
dataset$PAY_6 <-as.factor(dataset$PAY_6)

# Alterando a variavel dependente para o tipo fator
dataset$default.payment.next.month <- as.factor(dataset$default.payment.next.month)
head(dataset)
str(dataset)

# Renomeando a coluna de classe
colnames(dataset)
colnames(dataset)[25] <- "inadimplente"
colnames(dataset)

# Verificando valores missing e removendo do dataset
sapply(dataset, function(x) sum(is.na(x)))
missmap(dataset, main = "Valores Missing Observados")
dataset <- na.omit(dataset)

# Removendo a primeira coluna ID
dataset$ID <- NULL

# Total de inadimplentes versus nao-inadimplentes
table(dataset$inadimplente)

# Plot da distribuicao usando ggplot
qplot(inadimplente, data = dataset, geom = "bar") + theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Atribuindo uma semente
set.seed(12345)

# Amostragem estratificada. Selecionando as linhas de acordo com a variable inadimplente como strata
TrainingDataIndex <- createDataPartition(dataset$inadimplente, p = 0.45, list = FALSE)
TrainingDataIndex

# Criando Dados de Treinamento como subconjunto do conjunto de dados com números de indice de linha 
# conforme identificado acima e todas as colunas
trainData <- dataset[TrainingDataIndex,]
table(trainData$inadimplente)

# Analisando porcentagens entre as classes
prop.table(table(trainData$inadimplente))

# Número de linhas no dataset de treinamento
nrow(trainData)

# Compara as porcentagens entre as classes de treinamento e dados originais
DistributionCompare <- cbind(prop.table(table(trainData$inadimplente)), prop.table(table(dataset$inadimplente)))
colnames(DistributionCompare) <- c("Treinamento", "Original")
DistributionCompare

# Melt Data - Converte colunas em linhas
meltedDComp <- melt(DistributionCompare)
meltedDComp

# Plotando para ver a distribuicao do treinamento vs original - eh representativo ou existe sobre / sob amostragem?
ggplot(meltedDComp, aes(x = X1, y = value)) + geom_bar( aes(fill = X2), stat = "identity", position = "dodge") + theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Tudo o que nao esta no dataset de treinamento esta no dataset de teste. Observe o sinal - (menos)
testData <- dataset[-TrainingDataIndex,]

# Usando uma validação cruzada de 10 folds para treinar e avaliar modelo
TrainingParameters <- trainControl(method = "cv", number = 10)


## Random Forest Classification Model ##

# Construindo o Modelo
rf_model <- randomForest(inadimplente ~ ., data = trainData)
rf_model

# Conferindo o erro do modelo
plot(rf_model, ylim = c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col = 1:3, fill = 1:3)

# importância das variaveis preditoras para as previsões
varImpPlot(rf_model)

# Obtendo as variáveis mais importantes
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Criando o rank de variaveis baseado na importancia
rankImportance <- varImportance %>% 
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Usando ggplot2 para visualizar a importancia relativa das variaveis
ggplot(rankImportance, aes(x = reorder(Variables, Importance), y = Importance, fill = Importance)) + 
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank), hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() 

# Previsões
predictionrf <- predict(rf_model, testData)

# Plotando Confusion Matrix
cmrf <- confusionMatrix(predictionrf, testData$inadimplente, positive = "1")
cmrf

# Salvando o modelo
saveRDS(rf_model, file = "rf_model.rds")

# Carregando o modelo
modelo <- readRDS("rf_model.rds")

# Calculando Precision, Recall e F1-Score, que são métricas de avaliaão do modelo preditivo
y <- testData$inadimplente
predictions <- predictionrf

precision <- posPredValue(predictions, y)
precision

recall <- sensitivity(predictions, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1


# O modelo desenvolvido obteve 81% de acurácia. Através da matriz de confusão o modelo 
# preditivo previu: 11931 inadimplentes e acertou, 2307 inadimplentes e errou, 743 adimplentes
# e arrou e por fim 1328 adimplentes e acertou em sua previsão.

# Essa foi uma pequena amostra de predição de dados para inadimplência de clientes usando R.



