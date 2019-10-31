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
    n = min(c( min( s1$get_n() ), min(s2$get_n())) ): max(c( max(s1$get_n() ), max(s2$get_n())) )
    return(n)
}
   
# =============================================================================================================================
# Main class
# =============================================================================================================================

# main class
Signal <- R6::R6Class("Signal", public = list(
    # constructor
    initialize = function(x, n, ...) {
        # check n and x is numeric and they have the same length
        stopifnot(is.numeric(x), is.numeric(n))
        stopifnot(length(x) ==  length(n))

        # set them
        private$n <- n
        private$x <- x
    },

    # get n function
    get_n = function(...) {
        return(private$n)
    },

    # prin function
    print = function(...) {
        print("Signal:")
        print("x:")
        print(private$x)
        print("n:")
        print(private$n)
    }

    ),

    private = list(
        x = 1:10,
        n = 1:10
    )

)