+-+-+-+-+ +-+-+-+-+-+-+-+
|P|r|e|m| |P|a|n|d|i|a|n|
+-+-+-+-+ +-+-+-+-+-+-+-+
::::::::::::::::::::::::::::::::::::::::
Generalized Linear Model (glmnet)
Package: glmnet
::::::::::::::::::::::::::::::::::::::::

# Load Libraries
library(glmnet)

# Working Directory
setwd("~/Dropbox/Kaggle/avsc/processed/")

# Load Files
csvTrainfile <- "vw_train.csv"
csvTestfile <- "vw_test.csv"
testHistoryfile <- "testHistory.csv"

testHistory <- read.csv(testHistoryfile)
numTest <- nrow(testHistory)
training <- read.csv(csvTrainfile)

# Pre Process
target <- training$repeter
training[is.na(training)] <- 0
xTrain <- as.matrix(training[, -(1:3)])

test <- read.csv(csvTestfile)
test[is.na(test)] <- 0
xTest <- as.matrix(test[, -(1:3)])

# Run Model
model <- glmnet(xTrain, target, family = "binomial", alpha=0,lambda = 2^30)
cv.out <- cv.glmnet(xTrain, target, family = "binomial", alpha = 0)


# The predictions for IDs not existing in test.csv are 0.
pred <- rep(0, numTest)
pred[testHistory$id %in% test$id] <- predict(model, xTest, type = "response")

# Write output
write.table(data.frame(id = testHistory$id, repeatProbability = pred), 'pred.csv',
            sep=",", quote = FALSE, row.names=FALSE)
