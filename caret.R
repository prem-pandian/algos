+-+-+-+-+ +-+-+-+-+-+-+-+
|P|r|e|m| |P|a|n|d|i|a|n|
+-+-+-+-+ +-+-+-+-+-+-+-+
  
::::::::::::::::::::::::::::::::::::::::  
Regression Models using CARET package
::::::::::::::::::::::::::::::::::::::::
Five Different Models
Models:
1. KNN3
2. LDA
3. NNET
4. NB
5. RPART
6. SVM
Package: caret
::::::::::::::::::::::::::::::::::::::::

#Install Packages
install.packages(c("doMC","caret", "klaR", "nnet", "rpart", "e1071"))

#Load Libraries
library(doMC)
library(caret)
library(MASS)
library(klaR)
library(nnet)
library(e1071)
#library(rpart)

#register multi-core backend
registerDoMC(cores=4)

#how many workers do we have?
getDoParWorkers()

#we use the iris data set (we shuffle it first)
qTrain <- read.csv(file="TrainMerged_out.csv")

#Factor Variables
qTrainMerged <- qTrain[,c(1,2,4,5,6,7,8,9,10,11,12,13,15,16,17,18,3)]
qTrainMerged$Store <- as.factor(qTrainMerged$Store)
qTrainMerged$Dept <- as.factor(qTrainMerged$Dept)
qTrainMerged$IsHoliday <- as.factor(qTrainMerged$IsHoliday)
qTrainMerged$Month <- as.factor(qTrainMerged$Month)
qTrainMerged$Week <- as.factor(qTrainMerged$Week)

x <- qTrainMerged

#helper function to calculate the missclassification rate
posteriorToClass <- function(predicted) {
  colnames(predicted$posterior)[apply(predicted$posterior, 
                                      MARGIN=1, FUN=function(x) which.max(x))]
}

missclassRate <- function(predicted, true) {
  confusionM <- table(true, predicted)
  n <- length(true)
  
  tp <- sum(diag(confusionM))
  (n - tp)/n
}


#evaluation function which randomly selects 10% for testing
#and the rest for training and then creates and evaluates
#all models.
evaluation <- function() {
  #10% for testing
  testSize <- floor(nrow(x) * 10/100)
  test <- sample(1:nrow(x), testSize)
  
  train_data <- x[-test,]
  test_data <- x[test, -5]
  test_class <- x[test, 5]
  
  #create model
 model_knn3 <- knn3(Species~., data=train_data)
 model_lda <- lda(Species~., data=train_data)
 model_nnet <- nnet(Species~., data=train_data, size=10)
 model_nb <- NaiveBayes(Species~., data=train_data)
  model_svm <- svm(Weekly_Sales ~., data=train_data)
 model_rpart <- rpart(Species~., data=train_data)
  
  #prediction
predicted_knn3 <- predict(model_knn3 , test_data, type="class")
 predicted_lda <- posteriorToClass(predict(model_lda , test_data))
 predicted_nnet <- predict(model_nnet, test_data, type="class")
 predicted_nb <- posteriorToClass(predict(model_nb, test_data))
  predicted_svm <- predict(model_svm, test_data)
 predicted_rpart <- predict(model_rpart, test_data, type="class")

predicted <- list(svm=predicted_svm)

 predicted <- list(knn3=predicted_knn3, lda=predicted_lda, 
                   nnet=predicted_nnet, nb=predicted_nb, svm=predicted_svm, 
                   rpart=predicted_rpart)
  
  #calculate missclassifiaction rate
  sapply(predicted, FUN=
           function(x) missclassRate(true= test_class, predicted=x))
}


#now we run the evaluation
runs <- 1

#run parallel on all cores (with %dopar%)
ptime <- system.time({
  pr <- foreach(1:runs, .combine = rbind) %dopar% evaluation()
})

#compare results
r <- rbind(parallel=colMeans(pr))

#plot results
cols <- gray(c(.4,.8))
barplot(r, beside=TRUE, main="Avg. Miss-Classification Rate", col=cols)
legend("topleft", rownames(r), col=cols, pch=15)
