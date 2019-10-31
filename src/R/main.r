
source("signal.r")

sig <- Signal$new(c(1, 2, 3, 2, 4, 1), 1:6)
sig$print()
sig2 <- Signal$new(c(1, 4, 2, 3, 2, 1), 5:10)
sig2$print()

new_n <- .equalize_range(sig, sig2)

print(new_n)
