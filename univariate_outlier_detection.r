# keywords: seasonality, trend, noise, residual, moving average, movin median, robust, anomaly, outlier

# Install and import TSA (Time Series Analysis) package
# install.packages("TSA")
library(TSA)

# read the Google Analaytics PageView report
raw = read.csv("Downloads/20131120-20151110-google-analytics.csv")

# compute the Fourier Transform
fourier = periodogram(raw$Visite)

dd = data.frame(freq=fourier$freq, spec=fourier$spec)
order = dd[order(-dd$spec),]
top2 = head(order, 2)

# display the 2 highest "power" frequencies
top2

# convert frequency to time periods
seasons = 1/top2$f
seasons


set.seed(4)
data <- read.csv("Downloads/20131120-20151110-google-analytics.csv", sep = ",", header = T)
days = as.numeric(data$Visite)
cat("Days: ", days)

# insert some outliers ^1.2
for (i in 1:45 ) {
  pos = floor(runif(1, 1, 50))
  days[i*15+pos] = days[i*15+pos]^1.2
}

days[510+pos] = 0
plot(as.ts(days))

# install required library
install.packages("FBN")
library(FBN)

decomposed_days = decompose(ts(days, frequency = 7), "multiplicative")
plot(decomposed_days)


# Using Normal Distribution to Find Minimum and Maximum
random = decomposed_days$random
min = mean(random, na.rm = T) - 4*sd(random, na.rm = T)
max = mean(random, na.rm = T) + 4*sd(random, na.rm = T)

plot(as.ts(as.vector(random)), ylim = c(-0.5,2.5))
abline(h=max, col="#e15f3f", lwd=2)
abline(h=min, col="#e15f3f", lwd=2)


# find anomaly based on moving average
position = data.frame(id=seq(1, length(random)), value=random)
anomalyH = position[position$value > max, ]
anomalyH = anomalyH[!is.na(anomalyH$value), ]
anomalyL = position[position$value < min, ]
anomalyL = anomalyL[!is.na(anomalyL$value), ]
anomaly = data.frame(id=c(anomalyH$id, anomalyL$id), value=c(anomalyH$value, anomalyL$value))
anomaly = anomaly[!is.na(anomaly$value), ]

plot(as.ts(days))
real = data.frame(id=seq(1, length(days)), value=days)
realAnomaly = real[anomaly$id, ]
points(x = realAnomaly$id, y =realAnomaly$value, col="#e15f3f")


#install library
install.packages("forecast")
library(forecast)
library(stats)

trend = runmed(days, 7)
plot(as.ts(trend))


# Using the Normal Distribution to Find the Minimum and Maximum
detrend = days / as.vector(trend)
m = t(matrix(data = detrend, nrow = 7))
seasonal = colMeans(m, na.rm = T)
random = days / (trend * seasonal)
rm_random = runmed(random[!is.na(random)], 3)

min = mean(rm_random, na.rm = T) - 4*sd(rm_random, na.rm = T)
max = mean(rm_random, na.rm = T) + 4*sd(rm_random, na.rm = T)
plot(as.ts(random))
abline(h=max, col="#e15f3f", lwd=2)
abline(h=min, col="#e15f3f", lwd=2)


# find anomaly based on moving median
position = data.frame(id=seq(1, length(random)), value=random)
anomalyH = position[position$value > max, ]
anomalyH = anomalyH[!is.na(anomalyH$value), ]
anomalyL = position[position$value < min, ]
anomalyL = anomalyL[!is.na(anomalyL$value)]
anomaly = data.frame(id=c(anomalyH$id, anomalyL$id),
                     value=c(anomalyH$value, anomalyL$value))
points(x = anomaly$id, y =anomaly$value, col="#e15f3f")

plot(as.ts(days))
real = data.frame(id=seq(1, length(days)), value=days)
realAnomaly = real[anomaly$id, ]
points(x = realAnomaly$id, y =realAnomaly$value, col="#e15f3f")