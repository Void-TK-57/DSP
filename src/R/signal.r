# Main Signal Library for Digital Signal Processing
# Author: Void-TK-57

# add libraries
library(R6)

# =============================================================================================================================
# Auxiliary Functions
# =============================================================================================================================

# method do equalize range
.equalize_range <- function(s1, s2) {
    # equalize n
    n <- min(c( min( s1$get_n() ), min(s2$get_n())) ): max(c( max(s1$get_n() ), max(s2$get_n())) )
    # create vector of zeros
    x1 <- numeric(length(n))
    x2 <- numeric(length(n))
    
    # change x1 and x2
    x1[n >= min(s1$get_n()) & n <= max(s1$get_n())] <- s1$get_x()
    x2[n >= min(s2$get_n()) & n <= max(s2$get_n())] <- s2$get_x()
    return( list(n=n, x1=x1, x2=x2) )
}

# method to do a sum
.sum <- function(s1, s2) {
    # equalize range
    equalized <- .equalize_range(s1, s2)
    # get sum of x
    x <- equalized$x1 + equalized$x2
    # create new signal
    signal <- Signal$new(x, equalized$n)
    # return signal
    return(signal)
}

# method to do a sub
.sub <- function(s1, s2) {
    # equalize range
    equalized <- .equalize_range(s1, s2)
    # get sum of x
    x <- equalized$x1 - equalized$x2
    # create new signal
    signal <- Signal$new(x, equalized$n)
    # return signal
    return(signal)
}

# method to do a mul
.mul <- function(s1, s2) {
    # equalize range
    equalized <- .equalize_range(s1, s2)
    # get sum of x
    x <- equalized$x1 * equalized$x2
    # create new signal
    signal <- Signal$new(x, equalized$n)
    # return signal
    return(signal)
}

# method to do a div
.div <- function(s1, s2) {
    # equalize range
    equalized <- .equalize_range(s1, s2)
    # get sum of x
    x <- equalized$x1 / equalized$x2
    # create new signal
    signal <- Signal$new(x, equalized$n)
    # return signal
    return(signal)
}

# method to do a exp
.exp <- function(s1, s2) {
    # equalize range
    equalized <- .equalize_range(s1, s2)
    # get sum of x
    x <- equalized$x1 ** equalized$x2
    # create new signal
    signal <- Signal$new(x, equalized$n)
    # return signal
    return(signal)
}



   
# =============================================================================================================================
# Main class
# =============================================================================================================================

# main class
Signal <- R6::R6Class( c("Signal", "IAddable"), public = list(
    # constructor
    initialize = function(x, n, ...) {
        # check n and x is numeric and they have the same length
        stopifnot(is.numeric(n))
        stopifnot(length(x) ==  length(n))

        # set them
        private$n <- n
        private$x <- x
    },

    # get n function
    get_n = function(...) {
        return(private$n)
    },

    # get x function
    get_x = function(...) {
        return(private$x)
    },

    # prin function
    print = function(...) {
        print("Signal:")
        print("x:")
        print(private$x)
        print("n:")
        print(private$n)
    },

    plot = function(name = "singal", format = "jpg", local = "../../data", main="Signal", ylab="y", type="p", col="red") {
        path <- paste(local, "/", name, ".", format, sep="")
        jpeg(path)
        plot(private$n, private$x, main=main, ylab=ylab, type=type, col=col)
        dev.off()
    }

    ),

    private = list(
        x = 1:10,
        n = 1:10
    )

)

# =============================================================================================================================
# Overload Operators
# =============================================================================================================================

# overload + operator
`+.IAddable` = function(s1,s2) {
    if (is.numeric(s2) || is.integer(s2) || is.complex(s2) ) {
        # copy x and n
        x <- s1$get_x()
        n <- s1$get_n()
        # add to x
        x <- x + s2
        # return new signal
        return(Signal$new(x, n))
    } else{
        # return +
        return(.sum(s1, s2))
    }
}

# overload - operator
`-.IAddable` = function(s1,s2) {
    if (is.numeric(s2) || is.integer(s2) || is.complex(s2) ) {
        # copy x and n
        x <- s1$get_x()
        n <- s1$get_n()
        # add to x
        x <- x - s2
        # return new signal
        return(Signal$new(x, n))
    } else{
        # return -
        return(.sub(s1, s2))
    }
}

# overload * operator
`*.IAddable` = function(s1,s2) {
    if (is.numeric(s2) || is.integer(s2) || is.complex(s2) ) {
        # copy x and n
        x <- s1$get_x()
        n <- s1$get_n()
        # add to x
        x <- x * s2
        # return new signal
        return(Signal$new(x, n))
    } else{
        # return *
        return(.mul(s1, s2))
    }
}

# overload / operator
`/.IAddable` = function(s1,s2) {
    if (is.numeric(s2) || is.integer(s2) || is.complex(s2) ) {
        # copy x and n
        x <- s1$get_x()
        n <- s1$get_n()
        # add to x
        x <- x / s2
        # return new signal
        return(Signal$new(x, n))
    } else{
        # return /
        return(.div(s1, s2))
    }
}

# overload ** operator
`^.IAddable` = function(s1,s2) {
    if (is.numeric(s2) || is.integer(s2) || is.complex(s2) ) {
        # copy x and n
        x <- s1$get_x()
        n <- s1$get_n()
        # add to x
        x <- x ** s2
        # return new signal
        return(Signal$new(x, n))
    } else{
        # return **
        return(.exp(s1, s2))
    }
}

# =============================================================================================================================
# Functions for Base Signals
# =============================================================================================================================



# function to create a unit step signal
unit_step <- function(n0 = 0, n = 0:10) {
    # create vector os zeros
    x <- rep(0, length(n))
    # change value from n0 to the end to 1
    x[n >= n0] <- 1
    # create new signal from these values
    return(Signal$new( x, n) )
}

# function to create a unit sample signal
unit_sample <- function(n0 = 0, n = 0:10) {
    # create vector os zeros
    x <- rep(0, length(n))
    # change value from n0 to the end to 1
    x[n == n0] <- 1
    # create new signal from these values
    return(Signal$new( x, n) )
}

# function to create sinusoid
sinusoid <- function(amplitude = 1, teta = 0, omega = pi, n = 0:10) {
    # create x based on sin function apply to domain x
    x <- amplitude*sin(teta + omega*n)
    # create new signal from these values
    return(Signal$new( x, n) )  
}

# function to create real exponential
real_exp <- function(base = 1, n = 0:10) {
    # create x based on exp apply to domain x
    x <- base**n
    # create new signal from these values
    return(Signal$new( x, n) )  
}

# function to create complex exponential
complex_exp <- function(absolute = 1, teta = 0, omega = pi, n = 0:10) {
    # create x based on e to the power of the domain n
    x = absolute*exp( ((teta + omega)*1i)*n )
    # create new signal from these values
    return(Signal$new( x, n) )
}
