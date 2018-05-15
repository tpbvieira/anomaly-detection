library(rTensor)

space_index <- seq(-1, 1, l = 100)
bell_curve  <- dnorm(space_index, mean = 0, sd = 0.5)

case1 <- matrix(rep(bell_curve, 10), 100, 100)
case2 <- matrix(rep(bell_curve, 10), 100, 100)
case3 <- matrix(rep(bell_curve, 10), 100, 100)
case2[ , 51:100] <- case2[ , 51:100] + 0.1
case3[ , 51:100] <- case3[ , 51:100] - 0.1

X <- array(NA, dim = c(90, 100, 100))

for(i in 1:30) {
	X[i, , ]    <- case1 + matrix(rnorm(10000, sd = 0.1), 100, 100)
	X[i+30, , ] <- case2 + matrix(rnorm(10000, sd = 0.1), 100, 100)
	X[i+60, , ] <- case3 + matrix(rnorm(10000, sd = 0.1), 100, 100)
}

dim(X)

cp_decomp <- cp(as.tensor(X), num_components = 1)

str(cp_decomp)
# List of 3
#  $ : num [1:90, 1] 0.0111 0.0111 0.0111 0.0111 0.0112 ...
#  $ : num [1:100, 1] -0.00233 -0.00251 -0.00271 -0.00292 -0.00314 ...
#  $ : num [1:100, 1] -0.00996 -0.00994 -0.00996 -0.00993 -0.00997 ...
# NULL