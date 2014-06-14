+-+-+-+-+ +-+-+-+-+-+-+-+
|P|r|e|m| |P|a|n|d|i|a|n|
+-+-+-+-+ +-+-+-+-+-+-+-+

::::::::::::::::::::::::::::::::::::::::
Ensemble Model
Models:Random Forest, SVM, GBM, BayesGLM
::::::::::::::::::::::::::::::::::::::::

# Install Libraries  
install.packages(c("arm","caret","gbm","randomForest","caTools","foreach","doMC","e1071","SnowballC"))

  
# Load Libraries  
library(arm)
library(caret)
library(gbm)
library(randomForest)
library(caTools)
library(foreach)
library(doMC)
library(e1071)
library(SnowballC)

registerDoMC(cores=32)

#Calc AUC
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


#Build RF Models
buildRFModel <- function(training, pctDeadbeat) {
  gc(reset=TRUE)
  
  cat("inside buildRFModel \n")
  
  k <- 50 # ntree parameter
  l <- 8 # ntree parameter
  
  m <- 1 # id variable position
  p <- 42 # label variable position
  
  cat("\n**************\n\nRF pctDeadbeat=",pctDeadbeat,"\n\n***********\n\n")
  RF <- foreach(ntree=rep(k,l), .combine=combine,
                .multicombine = TRUE,
                .packages="randomForest") %dopar% {
                  training.label <- training$label
                  training <- training[,-c(m,p)]
                  classwt <- c((1-pctDeadbeat)/sum(training.label == 1),
                               pctDeadbeat/sum(training.label == 0)) *
                    nrow(training)
                  
                  randomForest(training,
                               factor(training.label),
                               ntree=ntree,
                               strata=factor(training.label),
                               do.trace=TRUE, importance=TRUE, forest=TRUE,
                               replace=TRUE, classwt=classwt)
                }
  RF
}

#Build RF Model Ensemble
buildRFModelEnsemble <- function(training) {
  cat("inside buildRFModelEnsemble \n")
  rfensemble<-lapply(list(rf1=0.25,
                          rf2=0.5,
                          rf3=0.75),
                     function(pctDeadbeat) buildRFModel(training, pctDeadbeat))  
  cat("exit buildRFModelEnsemble \n")
  rfensemble
}

#Build SVM Models
buildSVM <- function(training, cost=1) {
  
  cat("inside buildSVM \n")
  
  m <- 1 # id variable position
  p <- 42 # label variable position
  
  response <- factor(training$label)
  weight <- sum(training$label)/nrow(training)
  training <- training[,-c(m,p)]
  gc()
  weight <- c(1/(1-weight), 1/weight)
  names(weight) <- levels(response)
  
  cat("before SVM \n")
  
  svm(training, response, scale=FALSE, type='C-classification',
      kernel='radial', cachesize=4000, probability=TRUE,
      class.weights=weight, cost=cost)
}

#Build SVM Model Ensemble
buildSVMEnsemble <- function(training) {
  
  cat("inside buildSVMEnsemble \n")
  
  gc(reset=TRUE)
  mclapply(list(svm1=0.05, svm3=0.1, svm4=0.5, svm5=1),
           function(cost) buildSVM(training, cost=cost))
}

#Build GBM Models
buildGBMModel <- function(training, pctDeadbeat) {
  
  k <- 500 # ntree parameter
  l <- 10 # interaction dept
  
  cat("inside buildGBMModel \n")
  
  training$id <- NULL
  weight <- sum(training$label)/nrow(training)
  weights <- c((1-pctDeadbeat)/(1-weight),pctDeadbeat/weight)[1+training$label]
  GB <- gbm(training$label ~ ., data=training, n.trees=k,
            keep.data=FALSE, shrinkage=0.01, bag.fraction=0.3,
            weights = weights,
            interaction.depth=l, n.cores = 8)
  GB
}

#Build GBM Model Ensemble
buildGBMModelEnsemble <- function(training) {
  
  cat("inside buildGBMModelEnsemble \n")
  
  gc(reset=TRUE)
  mclapply(list(gb1=0.25,
                gb2=0.5,
                gb3=0.75),
           function(pctDeadbeat)
             buildGBMModel(training, pctDeadbeat))
}

#Build Bayes GLM Models
buildLinModel <- function(training) {
  
  k <- 1000 # niter parameter
  cat("inside buildLinModel \n")
  bayesglm(label ~ .,
           data = training[,-1], prior.scale=0.1,
           family=binomial, scaled=FALSE,
           n.iter=k)
}

#Build Bayes GLM Ensemble
buildLinEnsemble <- function(GBs,RFs,lin,training,preprocess) {
  submodels<-list(rfs=RFs,gbs=GBs,
                  #svms=SVMs,
                  lin=lin,
                  preprocess=preprocess)
  
  cat("before predictSubModels \n")
  z <- predictSubModels(submodels, training)
  z$label <- training$label
  cat("after predictSubModels \n")
  cat("Fitting final ensemble\n")
  submodels$ensemble<-bayesglm(label ~ ., family=binomial, data=z,
                               prior.df=Inf)
  cat("complete fitting ensemble\n")
  
  class(submodels) <- 'avsc'
  submodels
}

#Build SubModels
buildSubModels <- function(training, testing) {
  cat("Centering and scaling\n")
  preprocess <- preProcess(rbind(testing,training))
  
  training.label <- training$label
  training <- predict(preprocess, training)
  print(summary(training$label <- training.label))
  training.label <- NULL
  
  testing.label <- testing$label
  testing <- predict(preprocess, testing)
  testing$label <- testing.label
  testing.label <- NULL
  
  cat("Building models\n")
  gc(reset=TRUE)
  
  submodels <- lapply(list(rfs=buildRFModelEnsemble,
                           gbs=buildGBMModelEnsemble,
                           #svms=buildSVMEnsemble,
                           lin=buildLinModel),
                      function(f) f(training))
  cat("Complete Building models\n")
  
  submodels$preprocess <- preprocess
  submodels
}

#Build Models
buildModels <- function(submodels, testing) {
  
  cat("Inside buildModels\n")
  testing.label <- testing$label
  testing <- predict(submodels$preprocess, testing)
  testing$label <- testing.label
  
  cat("buildLinEnsemble start\n")  
  r <- buildLinEnsemble(submodels$gbs, submodels$rfs, #submodels$svms,
                        submodels$lin, testing,
                        submodels$preprocess)
  cat("buildLinEnsemble end\n")
  
  print(r)
  
  r
}

#Print function
print.avsc <- function(m) print(m$ensemble)

#Predict SubModels
predictSubModels <- function(model, d) {
  
  cat("inside predictSubModels\n")
  
  m <- 1 # id variable position
  p <- 42 # label variable position
  
  k <- 500 # ntree parameter
  
  d.label <- d$label
  d <- predict(model$preprocess, d)
  d$label <- d.label
  
  cat("predictSubModels start\n")
  gc()
  gbs <- lapply(model$gbs, function(subm) 1/(1+exp(-predict(subm, d[,-c(m,p)], n.tree=k))))
  gc()
  rfs <- lapply(model$rfs, function(subm) predict(subm, d[,-c(m,p)], type='prob')[,p])
  gc()
  #svms <- mclapply(model$svms, function(subm) attr(predict(subm, testing[1:100,-c(m,p)], probability=TRUE),"probabilities")[,1])
  
  cat("predictSubModels end\n")
  
  cbind(data.frame(gbs),
        data.frame(rfs),
        #data.frame(svms),
        lin=1/(1+exp(-predict(model$lin, d[,-c(m,p)]))))
}

#Predict Ensemble
predict.avsc <- function(m, d) {
  
  cat("inside predict.avsc\n")
  
  z <- predictSubModels(m, d)
  z$ensemble <- predict(m$ensemble, z, type='response')
  
  cat("exit predict.avsc\n")
  
  z
}

#Evaluate Results
evalResults.avsc <- function(m, training, test) {  
  trainTarget <- training$label
  train_pred <-predict(m, training)
  testTarget <- test$label
  test_pred <- predict(m, test)
  display_results(train_pred, trainTarget, test_pred, testTarget)
}

#Load Data
getData <- function() {
  training1 <- read.csv("final_train.csv")
  training <- training1[c(1:150000),]
  test <- training1[c(150001:160067),]
  row.names(training) <- NULL
  row.names(test) <- NULL
  list(training=training,testing=test)
}

#Make Submission
makeSubmission <- function(RF) {
  
  m <- 1 # id variable position
  p <- 42 # label variable position
  
  test <- read.csv("final_test.csv")
  gc()
  pred <- data.frame(predict(RF,test[,-c(m,p)], type='prob')[,p])
  gc()
  
  write.csv(pred, file="PR001.csv")
}

#Make Submission
makeSubmission.avsc <- function(model) {
  
  m <- 1 # id variable position
  p <- 42 # label variable position
  
  test <- read.csv("final_test.csv")
  test$avg_price_item_50 <- NULL
  pred<-predict(model, test)
  pred<-data.frame(Row_ID=test[,m],ensemble=pred$ensemble)
  pred$ensemble<-1/(1+exp(-pred$ensemble))
  write.csv(pred, file="PRlin.csv", row.names=FALSE)
}

#Main method
doItAll <- function() {
  
  cat("Getting data\n")
  print(system.time(d <- getData()))
  
  cat("Building SubModels\n")
  print(system.time(
    subm <- buildSubModels(d$training, d$testing)
  ))
  
  cat("Building Models\n")
  m <- buildModels(subm, d$testing)
  
  cat("Model Results\n")
  model.results <- list(d=d, m=m)
  
  cat("Remove Variables\n")
  subm <- NULL
  m <- NULL
  d <- NULL
  gc()
  
  cat("Evaluate Results\n")
  evalResults.avsc(model.results$m, model.results$d$training,
                   model.results$d$testing)
  gc()
  
  cat("Making Submission\n")  
  print(system.time(makeSubmission.avsc(model.results$m)))
  
  
  gc()
  model.results
}


