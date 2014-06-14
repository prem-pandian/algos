+-+-+-+-+ +-+-+-+-+-+-+-+
|P|r|e|m| |P|a|n|d|i|a|n|
+-+-+-+-+ +-+-+-+-+-+-+-+
  
::::::::::::::::::::::::::::::::::::::::  
ARIMA Model
Package:forecast
::::::::::::::::::::::::::::::::::::::::

# Load Libraries
library(forecast)
library(lubridate)
library(xts)
library(reshape2)

# Set Working Directory
setwd("~/Dropbox/kaggle/input")

# Load Files
qTrainMerged <- read.csv(file='TrainMerged.csv')
qTestMerged <- read.csv(file='TestMerged.csv')

#### TRAIN DATA ####

## "Convert" Training data to time series
qTrainMerged.ts <- ts(qTrainMerged)

## "Create" a vector of Regressors from training data

xregTrain <- cbind(Size=qTrainMerged$Size,Temperature=qTrainMerged$Temperature,Fuel_Price=qTrainMerged$Fuel_Price,MarkDown1=qTrainMerged$MarkDown1,
                  MarkDown2=qTrainMerged$MarkDown2,MarkDown3=qTrainMerged$MarkDown3,MarkDown4=qTrainMerged$MarkDown4,MarkDown5=qTrainMerged$MarkDown5,CPI=qTrainMerged$CPI,Unemployment=qTrainMerged$Unemployment)
xregTrain <- cbind(IsHoliday=model.matrix(~as.factor(qTrainMerged$IsHoliday)),xregTrain)
xregTrain <- xregTrain[,-1]
xregTrain <- cbind(Week=model.matrix(~as.factor(qTrainMerged$Week)),xregTrain)
xregTrain <- xregTrain[,-1]
xregTrain <- cbind(Dept=model.matrix(~as.factor(qTrainMerged$Dept)),xregTrain)
xregTrain <- xregTrain[,-1]
xregTrain <- cbind(Key=model.matrix(~as.factor(qTrainMerged$Store)),xregTrain)
xregTrain <- xregTrain[,-1]

## "Regress" Training data time series
qTrainMerged.arima <- auto.arima (x = qTrainMerged.ts[,5], stationary=FALSE, seasonal=TRUE, xreg=xregTrain,
                                  test=c("kpss","adf","pp"), seasonal.test=c("ocsb","ch"),
                                  allowdrift=TRUE,ic=c("aicc","aic", "bic"), parallel=TRUE, num.cores= 4, 
                                  stepwise = FALSE,approximation = FALSE)

#### TEST DATA ####

## "Create" a vector of Regressors from test data

xregTest <- cbind(Size=qTestMerged$Size,Temperature=qTestMerged$Temperature,Fuel_Price=qTestMerged$Fuel_Price,MarkDown1=qTestMerged$MarkDown1,
                   MarkDown2=qTestMerged$MarkDown2,MarkDown3=qTestMerged$MarkDown3,MarkDown4=qTestMerged$MarkDown4,MarkDown5=qTestMerged$MarkDown5,CPI=qTestMerged$CPI,Unemployment=qTestMerged$Unemployment,Week=qTestMerged$Week)
xregTest <- cbind(IsHoliday=model.matrix(~as.factor(qTestMerged$IsHoliday)),xregTest)
xregTest <- xregTest[,-1]
xregTest <- cbind(Dept=model.matrix(~as.factor(qTestMerged$Dept)),xregTest)
xregTest <- xregTest[,-1]
xregTest <- cbind(Store=model.matrix(~as.factor(qTestMerged$Store)),xregTest)
xregTest <- xregTest[,-1]

#### FORECAST ####

## "Forecast" data
qTestMerged.fcst <- forecast (qTrainMerged.arima, xreg =  xregTest, interval ="predict")

#### WRITE OUTPUT ####

## Set Working Directory
setwd("~/Dropbox/kaggle/submit")

## Temp File to prepare for submission
qOutputTemp <- paste(qTestMerged$Store,qTestMerged$Dept,qTestMerged$Date,sep="_")

## Write output to file
write.csv(qTestMerged.fcst , 'fcst_output_1.csv', row.names=FALSE)
write.csv(qOutputTemp , 'qOutputTemp_1.csv', row.names=FALSE)

