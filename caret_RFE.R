+-+-+-+-+ +-+-+-+-+-+-+-+
|P|r|e|m| |P|a|n|d|i|a|n|
+-+-+-+-+ +-+-+-+-+-+-+-+
  
::::::::::::::::::::::::::::::::::::::::  
caret Package: RFE
::::::::::::::::::::::::::::::::::::::::
Package: caret
::::::::::::::::::::::::::::::::::::::::

install.packages(c("caret","foreach","glmnet","doMC"))

library(caret)
library(foreach)
library(glmnet)
library(doMC)

registerDoMC(cores=32)

getDoParWorkers()

csvTrainfile <- "final_train.csv"
csvTestfile <- "final_test.csv"
testHistoryfile <- "testHistory.csv"

testHistory <- read.csv(testHistoryfile)
numTest <- nrow(testHistory)

training <- read.csv(csvTrainfile)
training[is.na(training)] <- 0

test <- read.csv(csvTestfile)
test[is.na(test)] <- 0

n_test <- test[, -c(1,112)]
xTest <- as.matrix(n_test)

xTrain <- training

target <- xTrain$label
xTrain <- xTrain[, -c(1,112)]
xTrain <- as.matrix(xTrain)

MyRFEcontrol <- rfeControl(
  functions = caretFuncs,
  number = 50,
  rerank = FALSE,
  returnResamp = "final",
  saveDetails = FALSE,
  verbose = TRUE)

MyTrainControl <- trainControl(
  method = "repeatedCV",
  number=10,
  repeats=1,
  returnResamp = "all",
  classProbs = TRUE,
  summaryFunction=twoClassSummary
)

RFE <- train(xTrain,as.factor(target),
             method='glmnet',
           tuneGrid = expand.grid(.alpha = 0, .lambda = 2^25),
           maximize=TRUE
           )

pred <- rep(0, numTest)
pred <- predict(RFE, xTest, type = "raw")


## write a submission file. Leaderboard Public: 0.59565
write.table(data.frame(id = testHistory$id, repeatProbability = pred), 'pred.csv',
            sep=",", quote = FALSE, row.names=FALSE)

