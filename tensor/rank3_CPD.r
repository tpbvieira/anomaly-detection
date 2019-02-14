message("## load libraries")
library(dplyr)
library(tidyr)
library(ggplot2)
library(rTensor)

## declare function to convert case to data_frame
case123_to_df <- function(case123, i) {
  as_data_frame(case123) %>%
  mutate(space_index = space_index) %>%
  gather(time_index, Value, -space_index) %>%
  mutate(time_index = as.numeric(gsub("V", "", time_index))) %>%
  mutate(case = i)
}

## generate data
message("## generate data")
# bell-shaped (normal) spacial component with different means
space_index <- seq(-1, 1, l = 100)
case1 <- matrix(rep(dnorm(space_index, mean = 0,    sd = 0.3), 10), 100, 100)
case2 <- matrix(rep(dnorm(space_index, mean = 0.5,  sd = 0.3), 10), 100, 100)
case3 <- matrix(rep(dnorm(space_index, mean = -0.5, sd = 0.3), 10), 100, 100)
## visualize noisy case 1, case 2, and case 3 means
setEPS()
postscript("normal_and_mean.eps")
bind_rows(case123_to_df(case1, "case 1"), case123_to_df(case2, "case 2"), case123_to_df(case3, "case 3")) %>%
  ggplot(aes(y = space_index, x = time_index, fill = Value)) +
    geom_tile() +
    facet_wrap(~case, nrow = 1) +
    xlab("Time") + 
    ylab("Space") +
    theme(legend.position = "bottom")
# sine-shaped temporal component
sine_wave <- sin(seq(-4*pi, 4*pi, l = 100))
sine_mat  <- matrix(rep(sine_wave, each = 100), 100, 100)
# add sine form
case1 <- case1 + 0.3 * sine_mat
case2 <- case2 + 0.6 * sine_mat
case3 <- case3 + 0.9 * sine_mat
setEPS()
postscript("normal_mean_sine.eps")
bind_rows(case123_to_df(case1, "case 1"), case123_to_df(case2, "case 2"), case123_to_df(case3, "case 3")) %>%
  ggplot(aes(y = space_index, x = time_index, fill = Value)) +
    geom_tile() +
    facet_wrap(~case, nrow = 1) +
    xlab("Time") + 
    ylab("Space") +
    theme(legend.position = "bottom")
# shifts +0,1 and -0,1 in the temporal component
case2[ , 51:100] <- case2[ , 51:100] + 0.1
case3[ , 51:100] <- case3[ , 51:100] - 0.1
# replicate case 1-3 mean data and add noise, in order to obtain a sample for CP analysis. After, organize these data into a 3-way array
X <- array(NA, dim = c(90, 100, 100))
for(i in 1:30) {
  X[i, , ]    <- case1 + matrix(rnorm(10000, sd = 0.1), 100, 100)
  X[i+30, , ] <- case2 + matrix(rnorm(10000, sd = 0.1), 100, 100)
  X[i+60, , ] <- case3 + matrix(rnorm(10000, sd = 0.1), 100, 100)
}
message("# X dimensions: ")
dim(X)
## visualize noisy case 1, case 2, and case 3 means
message("## visualize noisy case 1, case 2, and case 3 means")
setEPS()
postscript("noisy.eps")
bind_rows(case123_to_df(case1, "case 1"), case123_to_df(case2, "case 2"), case123_to_df(case3, "case 3")) %>%
  ggplot(aes(y = space_index, x = time_index, fill = Value)) +
    geom_tile() +
    facet_wrap(~case, nrow = 1) +
    xlab("Time") + 
    ylab("Space") +
    theme(legend.position = "bottom")

## CP decomposition
message("## CP decomposition")
# decompose into 
message("# decompose")
cp_decomp <- cp(as.tensor(X), num_components = 3, max_iter = 100)
# check convergence status
message("# Convergence: ", cp_decomp$conv)
# structure of the decomposed matrices
message("# Decomposed matrices: ", str(cp_decomp$U))
# percentage of norm explained
message("# percentage of norm explained: ", cp_decomp$norm_percent)

## visualize estimated CP components
setEPS()
postscript("rank3_CPD.eps")
data_frame(component = c(rep("u[1]", 90), rep("u[2]", 90), rep("u[3]", 90),
                         rep("v[1]", 100), rep("v[2]", 100), rep("v[3]", 100),
                         rep("w[1]", 100), rep("w[2]", 100), rep("w[3]", 100)),
           value = c(cp_decomp$U[[1]][,1], cp_decomp$U[[1]][,2], cp_decomp$U[[1]][,3],
                     cp_decomp$U[[2]][,1], cp_decomp$U[[2]][,2], cp_decomp$U[[2]][,3],
                     cp_decomp$U[[3]][,1], cp_decomp$U[[3]][,2], cp_decomp$U[[3]][,3]),
           index = c(rep(1:90, 3), rep(space_index, 3), rep(1:100, 3))) %>%
  ggplot(aes(index, value)) + 
    geom_line() + 
    facet_wrap(~component, scales = "free", nrow = 3, labeller = labeller(component = label_parsed)) + 
    theme(axis.title = element_blank())
dev.off()