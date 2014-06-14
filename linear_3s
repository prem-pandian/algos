+-+-+-+-+ +-+-+-+-+-+-+-+
|P|r|e|m| |P|a|n|d|i|a|n|
+-+-+-+-+ +-+-+-+-+-+-+-+
  
::::::::::::::::::::::::::::::::::::::::  
Linear Regression alternatives 
::::::::::::::::::::::::::::::::::::::::
Three Different Models
Models:
1. TSLM
2. LMER
3. LM
::::::::::::::::::::::::::::::::::::::::

# Load Libraries
library(forecast)

# Load Files
qTrainMerged <- read.csv(file='TrainMerged.csv')
qTestMerged <- read.csv(file='TestMerged.csv')

qTrainMerged.ts <- ts(qTrainMerged)
qTestMerged.ts <- ts(qTestMerged)

## "Regress" Training data time series
qTrainMerged.lm1 <- tslm(Weekly_Sales ~ factor(Store)+factor(Dept)+
                         factor(Week)+Temperature+Fuel_Price+MarkDown1+MarkDown2+MarkDown3+MarkDown4+
                         MarkDown5+Unemployment, data=qTrainMerged.ts)

qTrainMerged.lm2 <- lmer(Weekly_Sales ~ factor(Store)+factor(Date)+factor(Dept)+factor(IsHoliday)+factor(Type)+
                         factor(Week)+Size+Temperature+Fuel_Price+MarkDown1+MarkDown2+MarkDown3+MarkDown4+
                         MarkDown5+CPI+Unemployment, data=qTrainMerged.ts)

qTrainMerged.lm3 <- tslm(Weekly_Sales ~ factor(Store)+factor(Date)+factor(Dept)+factor(IsHoliday)+factor(Type)+
                                                      factor(Week)+Size+Temperature+Fuel_Price+MarkDown1+MarkDown2+MarkDown3+MarkDown4+
                                                      MarkDown5+CPI+Unemployment, data=qTrainMerged.ts)

qTrainMerged.lm4 <- lm(Weekly_Sales ~ factor(Store)+factor(Date)+factor(Dept)+factor(IsHoliday)+factor(Type)+
                           factor(Week)+Size+Temperature+Fuel_Price+MarkDown1+MarkDown2+MarkDown3+MarkDown4+
                           MarkDown5+CPI+Unemployment, data=qTrainMerged.ts)


xregTest <- qTestMerged[,c("Store","Dept","Week","Temperature",
                           "Fuel_Price","MarkDown1","MarkDown2","MarkDown3","MarkDown4",
                           "MarkDown5","CPI","Unemployment")]

xregTest.ts <- ts(xregTest)

xregTest$Store <- factor(xregTest$Store)
xregTest$Dept <- factor(xregTest$Dept)
xregTest$Week <- factor(xregTest$Week)

fits.predict <- predict(qTrainMerged.lm1,newdata=xregTest)

# Set Working Directory
setwd("~/Dropbox/kaggle/submit")

write.csv(fits.predict,file="fcst_output.csv",row.names=FALSE)

