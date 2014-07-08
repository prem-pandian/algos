+-+-+-+-+ +-+-+-+-+-+-+-+
|P|r|e|m| |P|a|n|d|i|a|n|
+-+-+-+-+ +-+-+-+-+-+-+-+
  
::::::::::::::::::::::::::::::::::::::::  
 GBM Models
::::::::::::::::::::::::::::::::::::::::
1. GBM - Adaboost
2. GBM - Bernoulli
::::::::::::::::::::::::::::::::::::::::
  
  
  
# Install Packages
install.packages(c("arm","caret","gbm","randomForest","caTools","foreach","doMC","e1071","DMwR","SnowballC","rpart"))

# Load Libraries
library(DMwR)
library(arm)
library(caret)
library(gbm)
library(randomForest)
library(caTools)
library(foreach)
library(MASS)
library(SnowballC)
library(doMC)
library(e1071)
library(rpart)

registerDoMC(cores=32)

##############
# Load Data 
##############
getData <- function() {
  training1 <- read.csv("vw_train.csv")
  testing1 <- read.csv("vw_test.csv")
  training <- training1
  test <- testing1
  row.names(training) <- NULL
  row.names(test) <- NULL
  training[is.na(training)] <- 0
  test[is.na(test)] <- 0
  list(training=training,testing=test)
}

################
#GBM  -- adaboost
################
gbm_adaboost <- function(train) {
  
  train <- train[,-c(1)]
  
  gbm_adaboost_model <- gbm(
    repeter ~ ., 
    distribution = "adaboost", 
    data = train,
    n.trees = 10000,
    interaction.depth = 6,
    n.minobsinnode = 30,
    shrinkage = 0.05,
    bag.fraction = 0.2,
    train.fraction = 0.8,
    verbose = TRUE,
    n.cores = 32)
  
  gbm_adaboost_model
  
}

################
#GBM  -- bernoulli
################

gbm_bernoulli <- function(train) {
  
  train <- train[,-c(1)]
  
  gbm_bernoulli_model <- gbm(
    repeter ~ ., 
    distribution = "bernoulli", 
    data = train,
    n.trees = 10000,
    interaction.depth = 6,
    n.minobsinnode = 30,
    shrinkage = 0.05,
    bag.fraction = 0.2,
    train.fraction = 0.8,
    verbose = TRUE,
    n.cores = 32)
  
  gbm_bernoulli_model
  
}




doItAll <- function() {
  
  cat("Getting data\n")
  d <- getData()
  
  cat("Building gbm adaboost\n")
  gbm_adaboost_model <- gbm_adaboost(d$training)

  
  cat("Building gbm bernoulli\n")
  gbm_bernoulli_model <- gbm_bernoulli(d$training)

  cat("Predict\n")
  gbm_adaboost_pred <- predict (gbm_adaboost_model, d$test[,-c(1,2)], type = "response",n.trees=10000)
  gbm_bernoulli_pred <- predict (gbm_bernoulli_model, d$test[,-c(1,2)], type = "response",n.trees=10000)
  
  cat("Combine Predictions\n")
  predict <- cbind(d$test$id,gbm_adaboost_pred,gbm_bernoulli_pred)
  
  
  testHistoryfile <- "testHistory.csv"
  
  testHistory <- read.csv(testHistoryfile)
  numTest <- nrow(testHistory)
  
  
  ## The predictions for IDs not existing in test.csv are 0.
  pred <- rep(0, numTest)
  pred[testHistory$id %in% predict$id] <- predict
  
  write.table(data.frame(id = testHistory$id, repeatProbability = pred), 'pred.csv',
              sep=",", quote = FALSE, row.names=FALSE)
  
  
}


  
  

