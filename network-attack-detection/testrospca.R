library(rospca)
library(dplyr)

dataPath <- "/home/thiago/dev/anomaly-detection/network-attack-detection/data/synthetic/"
resultPath <- "/home/thiago/dev/anomaly-detection/network-attack-detection/output/synthetic/robpca/"
fileList <- list.files(dataPath, pattern=glob2rx("*aussian_2400_0.33.csv"))

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
