library(rospca)
library(dplyr)

# Simulated Data Set
dataPath <- "data/synthetic/"
resultPath <- "output/synthetic/robpca/"
c = 0.01
for(i in 1:50) {
	gaussianFilePath <- paste(dataPath, "gaussian_2400_  ", format(round(c, 2), nsmall=2), ".csv", sep="")
	paretoFilePath <- paste(dataPath, "pareto_2400_  ", format(round(c, 2), nsmall=2), ".csv", sep="")
	lognormalFilePath <- paste(dataPath, "lognormal_2400_  ", format(round(c, 2), nsmall=2), ".csv", sep="")
	uniformCFilePath <- paste(dataPath, "uniform_c_2400_  ", format(round(c, 2), nsmall=2), ".csv", sep="")
	gaussianCFilePath <- paste(dataPath, "gaussian_c_2400_  ", format(round(c, 2), nsmall=2), ".csv", sep="")
	XguFilePath <- paste(resultPath, "robpca_k2_gaussian_2400_", format(round(c, 2), nsmall=2), ".csv", sep="")
	XpgFilePath <- paste(resultPath, "robpca_k2_pareto_2400_", format(round(c, 2), nsmall=2), ".csv", sep="")
	XlgFilePath <- paste(resultPath, "robpca_k2_lognormal_2400_", format(round(c, 2), nsmall=2), ".csv", sep="")

	# Gaussian with uniform anomalies
	if (!file.exists(XguFilePath)){
		gaussian_df <- read.csv(gaussianFilePath, header=TRUE, sep=",")
		gaussian_size = dim(gaussian_df)[1]
		gt_end = gaussian_size %/% 2
		gaussian = gaussian_df[1:gt_end,]		
		
		uniform_c_df <- read.csv(uniformCFilePath, header=TRUE, sep=",")

		Xgu = bind_rows(gaussian,uniform_c_df)

		print(c)
		resRS <- robpca(Xgu, k=2, skew=TRUE, ndir=5000)
		
		print(XguFilePath)
		write(resRS$flag.all, file=XguFilePath, sep = "\n")	
	}

	# Pareto with gaussian anomalies
	if (!file.exists(XpgFilePath)){
		pareto_df <- read.csv(paretoFilePath, header=TRUE, sep=",")		
		gaussian_c_df <- read.csv(gaussianCFilePath, header=TRUE, sep=",")

		Xpg = bind_rows(pareto_df,gaussian_c_df)

		print(c)
		resRS <- robpca(Xpg, k=2, skew=TRUE, ndir=5000)
		
		print(XpgFilePath)
		write(resRS$flag.all, file=XpgFilePath, sep = "\n")
	}

	# Lognormal with gaussian anomalies
	if (!file.exists(XlgFilePath)){
		lognormal_df <- read.csv(lognormalFilePath, header=TRUE, sep=",")		
		gaussian_c_df <- read.csv(gaussianCFilePath, header=TRUE, sep=",")

		Xpg = bind_rows(lognormal_df,gaussian_c_df)

		print(c)
		resRS <- robpca(Xpg, k=2, skew=TRUE, ndir=5000)
		
		print(XlgFilePath)
		write(resRS$flag.all, file=XlgFilePath, sep = "\n")
	}
	
	c = c + 0.01
}

# CTU-13 Data Set
dataPath <- "data/ctu_13/raw_clean_test_robpca_csv/33/data/"
resultPath <- "output/ctu_13/results/robpca/33/"
fileList <- list.files(dataPath, pattern=glob2rx("*.binetflow'"))
fileList <- list.files(dataPath, pattern=glob2rx("*.binetflow'"))
for(fileName in fileList){
	dataFilePath  <- paste(dataPath, fileName, sep="")
	print(dataFilePath)
	test_df <- read.csv(dataFilePath, header=TRUE, sep=",")	
	test_df <- test_df[,c("State", "dTos", "Dport", "Sport", "TotPkts", "TotBytes", "SrcBytes")]
	resRS <- robpca(test_df, k=2, skew=TRUE,ndir=5000)
	resultFilePath  <- paste(resultPath, fileName, sep="")
	print(resultFilePath)
	write(resRS$flag.all, file=resultFilePath, sep = "\n")
}
