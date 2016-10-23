start.time <- Sys.time()
# reads the files:
options(warn = -1)
kdd_train=read.csv(file="KDDTrain+.txt", sep = ",")
kdd_train=kdd_train[,-43]
# reads the names of the columns
colnames <- read.table("names", skip = 1, sep = ":")
# Sets the names on the trainingset
names(kdd_train) <- colnames$V1
# requires/installs the packages
require(class)
# predicts with KNN
#knn(train = kdd_train, test = kdd_train, cl = class, k = 355)
kdd_train$type_attack <- 0
#kdd_train$class <- as.character(kdd_train$class)
# loops through and writes the correct class based on the subclass which is attacks
for(i in 1:nrow(kdd_train))
  if((kdd_train[i,42]=="smurf")|(kdd_train[i,42]=="neptune")|(kdd_train[i,42]=="back")|(kdd_train[i,42]=="teardrop")|(kdd_train[i,42]=="pod")|(kdd_train[i,42]=="land")){
    kdd_train[i,43]="DoS"
  }else if(kdd_train[i,42]=='normal'){
    kdd_train[i,43]="Normal"
  }else if((kdd_train[i,42]=="buffer_overflow")|(kdd_train[i,42]=="loadmodule")|(kdd_train[i,42]=="perl")|(kdd_train[i,42]=="rootkit")){
    kdd_train[i,43]="U2R"
  }else if( (kdd_train[i,42]=="ftp_write")|(kdd_train[i,42]=="guess_passwd")|(kdd_train[i,42]=="multihop")|(kdd_train[i,42]=="phf")|(kdd_train[i,42]=="imap")|(kdd_train[i,42]=="spy")|(kdd_train[i,42]=="warezclient")|(kdd_train[i,42]=="warezmaster")){
    kdd_train[i,43]="R2L"
  }else if((kdd_train[i,42]=="ipsweep")|(kdd_train[i,42]=="nmap")|(kdd_train[i,42]=="portsweep")|(kdd_train[i,42]=="satan")){
    kdd_train[i,43]="Probe"
  }


trainIndex <- createDataPartition(kdd_train$type_attack, p = .6, list = F)
kddTraining = kdd_train[trainIndex,]
kddTesting = kdd_train[-trainIndex,]
kddTest = kddTesting
kddTrainingTarget = as.factor(kddTraining$type_attack)
kddTraining=kddTraining[, -c(2,3,4,42,43)]
kddTesting=kddTesting[, -c(2,3,4,42,43)]



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

pred<-(knn(kddTraining, kddTesting, kddTrainingTarget, k = knn.cross$best.parameter))
table(pred,kddTest$type_attack)
predicted <- data.frame(Predictions=(pred))
predicted$Actual=c(kddTest$type_attack)

predicted$accuracy <- 0
predicted$Actual=as.factor(predicted$Actual)
for(i in 1:nrow(predicted))
  if(predicted[i,1]==predicted[i,2]){
    predicted[i,3]="1"
  }else if(predicted[i,1]!=predicted[i,2]){
    predicted[i,3]="0"
  }
plot(predicted$accuracy[0:300])
lines(predicted$accuracy)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
