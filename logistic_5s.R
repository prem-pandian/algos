+-+-+-+-+ +-+-+-+-+-+-+-+
|P|r|e|m| |P|a|n|d|i|a|n|
+-+-+-+-+ +-+-+-+-+-+-+-+
  
::::::::::::::::::::::::::::::::::::::::  
logistic Regression alternatives (0/1)
::::::::::::::::::::::::::::::::::::::::
Five Different Models
Models:
1. GBM - Adaboost
2. GBM - Bernoulli
3. Random Forest - Weighted
4. Random Forest - Simple
5. Rpart
6. Bayes GLM
7. LDA
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

registerDoMC(cores=8)

##############
# Load Data 
##############
getData <- function() {
  training1 <- read.csv("final_train.csv")
  testing1 <- read.csv("final_test.csv")
  training <- training1
  test <- testing1
  row.names(training) <- NULL
  row.names(test) <- NULL
  list(training=training,testing=test)
}

################
#GBM  -- adaboost
################
gbm_adaboost <- function(train) {
  
  train <- train[,-c(1)]
  
  gbm_adaboost_model <- gbm(
    label ~ ., 
    distribution = "adaboost", 
    data = train,
    n.trees = 5000,
    interaction.depth = 4,
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
    label ~ ., 
    distribution = "bernoulli", 
    data = train,
    n.trees = 5000,
    interaction.depth = 4,
    n.minobsinnode = 30,
    shrinkage = 0.05,
    bag.fraction = 0.2,
    train.fraction = 0.8,
    verbose = TRUE,
    n.cores = 32)
  
  gbm_bernoulli_model
  
}

################
# Random Forest with Weights
################

random_forest_wt <- function(train) {
  
  rf_model <- foreach(ntree=rep(70,32), .combine=combine,
                      .multicombine = TRUE,
                      .packages="randomForest") %dopar% {
                        train.label <- train$label
                        train <- train[,-c(1,41)]
                        classwt <- c(0.80/sum(train.label == 1),
                                     0.20/sum(train.label == 0)) *
                          nrow(train)
                        
                        randomForest(train,
                                     factor(train.label),
                                     ntree=ntree,
                                     strata=factor(train.label),
                                     do.trace=TRUE, importance=TRUE, forest=TRUE,
                                     replace=TRUE,classwt=classwt)
                      }
  rf_model
  
}


################
# Random Forest
################

random_forest <- function(train) {
  
  rf_model <- foreach(ntree=rep(70,32), .combine=combine,
                      .multicombine = TRUE,
                      .packages="randomForest") %dopar% {
                        train.label <- train$label
                        train <- train[,-c(1,41)]
                        
                        randomForest(train,
                                     factor(train.label),
                                     ntree=ntree,
                                     strata=factor(train.label),
                                     do.trace=TRUE, importance=TRUE, forest=TRUE,
                                     replace=TRUE)
                      }
  rf_model
  
}

################
#rpart
################

r_part <- function (train) {
  
  train <- train[,-c(1)]
  
  rpart_model <- train (label ~ ., 
                        train,
                        method = "rpart")
  
  rpart_model
  
}


################
#bayesglm 
################

bayes_glm <- function (train) {
  
  train <- train[,-c(1)]
  
  bayesglm_model <- bayesglm(label ~ .,
                             data=train, 
                             family=binomial)
  
  bayesglm_model
  
}



################
#lda 
################

ldamodel <- function (train) {
  
  train <- train[,-c(1)]
  
  lda_model <- lda(label ~ .,
                  data=train)
  
  lda_model
  
}

doItAll <- function() {
  
  cat("Getting data\n")
  print(system.time(d <- getData()))
  
  cat("Building gbm adaboost\n")
  print(system.time(
    gbm_adaboost_model <- gbm_adaboost(d$training)
  ))
  
  cat("Building gbm bernoulli\n")
  print(system.time(
    gbm_bernoulli_model <- gbm_bernoulli(d$training)
  ))
  
  cat("Building Random Forest with Weights\n")
  print(system.time(
    random_forest_model_wt <- random_forest_wt(d$training)
  ))
  
  cat("Building Random Forest\n")
  print(system.time(
    random_forest_model <- random_forest(d$training)
  ))
  
  cat("Building rpart\n")
  print(system.time(
    rpart_model <- r_part(d$training)
  ))
  
  cat("Building bayesglm\n")
  print(system.time(
    bayesglm_model <- bayes_glm(d$training)
  ))
  
  
  cat("Building lda\n")
  print(system.time(
    lda_model <- ldamodel(d$training)
  ))
  
  models <- list(adaboost = gbm_adaboost_model,
                 bernoulli = gbm_bernoulli_model,
                 lda=lda_model,
                 rf_wt = random_forest_model_wt,
                 rf=random_forest_model,
                 rpart=rpart_model)
  
  cat("Evaluate Results\n")
  x <- lapply(models, function (models)
        evalResults.avsc(models, d$training,
                   d$testing))
  
}  



#################
# Calculate AUC 
#################
display_results <- function(train_pred, trainTarget, test_pred, testTarget){
  
  cat("inside display_results \n")
  train_AUC <- colAUC(train_pred, trainTarget)
  test_AUC <- colAUC(test_pred, testTarget)
  cat("\n\n*** what ***\ntraining:")
  print(train_AUC)
  cat("\ntesting:")
  print(test_AUC)
  cat("\n*****************************\n")
  list(train_AUC=train_AUC,test_AUC=test_AUC)
}

#################
# Evaluate Results
#################
evalResults.avsc <- function(m, training, test) {
  trainTarget <- training$label
  train_pred <-predict (m, training, type = "response")
  
  testTarget <- test$label
  test_pred <- predict (m, test, type = "response")
  
  auc_results <- display_results(train_pred, trainTarget, test_pred, testTarget)
  auc_results
}


#################
# write output
#################

write.csv(predict, file = "predict.csv", quote = FALSE, row.names = FALSE)
