
source("signal.r")

sig <- Signal$new(c(1, 1, 1, 1, 1, 1), 1:6)

sig2 <- Signal$new(c(2, 2, 2, 2, 2, 2), 5:10)


print("==========")
new_n <- .sum(sig, sig2)
print(new_n)
new_n <- .sub(sig, sig2)
print(new_n)
new_n <- .mul(sig, sig2)
print(new_n)
new_n <- .div(sig, sig2)
print(new_n)
