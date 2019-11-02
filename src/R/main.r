
source("signal.r")

print("=========")

sig <- unit_step(5, n = 1:10)
sig2 <- unit_step(7, n = 3:12)


sig3 <- sig + 2

sig3 <- sig3 / 3

print(sig3)