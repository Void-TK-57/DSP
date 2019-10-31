
source("signal.r")

sig <- Signal$new(c(1, 1, 1, 1, 1, 1), 1:6)
sig$print()
sig2 <- Signal$new(c(2, 2, 2, 2, 2, 2), 5:10)
sig2$print()

print("==========")
new_n <- .equalize_range(sig, sig2)
print(new_n)
