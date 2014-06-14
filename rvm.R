+-+-+-+-+ +-+-+-+-+-+-+-+
|P|r|e|m| |P|a|n|d|i|a|n|
+-+-+-+-+ +-+-+-+-+-+-+-+
  
::::::::::::::::::::::::::::::::::::::::  
Vector Machine Regression Models
::::::::::::::::::::::::::::::::::::::::
Models: RVM
Package: kernlab
::::::::::::::::::::::::::::::::::::::::

# Load Libraries
library(forecast)
library(kernlab)
library(gbm)

# Load Files
qTrainMerged <- read.csv(file='TrainMerged.csv')
qTestMerged <- read.csv(file='TestMerged.csv')

qTrainMerged.ts <- ts(qTrainMerged)
qTestMerged.ts <- ts(qTestMerged)

## "Regress" Training data time series


  qTrainMerged.lm2 <- gbm(Weekly_Sales ~ factor(Dept)+factor(IsHoliday)+factor(Type)+
                           factor(Week)+Size+Temperature+Fuel_Price+MarkDown1+MarkDown2+MarkDown3+MarkDown4+
                           MarkDown5+CPI+Unemployment, data=qTrainMerged.ts , distribution='multinomial')

qTrainMerged.lm1 <- rvm(Weekly_Sales ~ factor(Store)+factor(Dept)+
                         factor(Week)+Temperature+Fuel_Price+MarkDown1+MarkDown2+MarkDown3+MarkDown4+
                         MarkDown5+Unemployment, data=qTrainMerged.ts,kernel = "vanilladot")

xregTest <- qTestMerged[,c("Store","Dept","Week","Temperature",
                           "Fuel_Price","MarkDown1","MarkDown2","MarkDown3","MarkDown4",
                           "MarkDown5","CPI","Unemployment")]

xregTest.ts <- ts(xregTest)

xregTest$Store <- factor(xregTest$Store)
xregTest$Dept <- factor(xregTest$Dept)
xregTest$Week <- factor(xregTest$Week)

fits.predict <- predict(qTrainMerged.lm1,newdata=xregTest)

# Set Working Directory
setwd("~/Dropbox/Kaggle/submit")

write.csv(fits.predict,file="fcst_output.csv",row.names=FALSE)

