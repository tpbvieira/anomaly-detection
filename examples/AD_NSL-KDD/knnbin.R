start.time <- Sys.time()
# reads the files:
options(warn = -1)
#kdd_train=read.csv(file="KDDTrain+.txt", sep = ",")
kdd_train=read.csv(file="KDDTrain+.arff", sep = ",")
#kdd_train=kdd_train[,-43]
# reads the names of the columns
colnames <- read.table("names", skip = 1, sep = ":")
# Sets the names on the trainingset
names(kdd_train) <- colnames$V1
# requires/installs the packages
require(class)
trainIndex <- createDataPartition(kdd_train$attacks, p = .6, list = F)
kddTraining = kdd_train[trainIndex,]
kddTesting = kdd_train[-trainIndex,]
kddTest = kddTesting
kddTrainingTarget = as.factor(kddTraining$attacks)
kddTraining=kddTraining[, -c(2,3,4,42)]
kddTesting=kddTesting[, -c(2,3,4,42)]


zeroVarianceFeatures <- sapply(kddTraining, function(i){
  if((is.numeric(i) & !any(is.nan(i)) & sd(i) >0) | is.factor(i) | is.character(i)) TRUE
  else FALSE
  
})

sapply(kddTraining, function(x)all(is.na(x)))
naValuesTest <-  function (x) {
  w <- sapply(x, function(x)all(is.na(x)))
  if (any(w)) {
    stop(paste("All NA values are found in columns", paste(which(w), collapse=", ")))
  }
}


naValuesTest(kddTraining)

knn.cross <- tune.knn(x = kddTraining, y = kddTrainingTarget, k = 1:20,tunecontrol=tune.control(sampling = "cross"), cross=10)

pred<-(knn(kddTraining, kddTesting, kddTrainingTarget, k = 3))
table(pred,kddTest$attacks)
predicted <- data.frame(Predictions=(pred))
predicted$Actual=c(as.character(kddTest$attacks))

predicted$accuracy <- 0
predicted$Actual=as.factor(predicted$Actual)
for(i in 1:nrow(predicted))
  if(predicted[i,1]==predicted[i,2]){
    predicted[i,3]="1"
  }else if(predicted[i,1]!=predicted[i,2]){
    predicted[i,3]="0"
  }
plot(as.numeric(predicted$accuracy[0:300]))
lines(predicted$accuracy)

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
