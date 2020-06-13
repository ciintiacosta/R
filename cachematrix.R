
## The makeCacheMatrix function creates a matrix object that 
##stores the inverse of the matrix in its cache

makeCacheMatrix <- function(x = matrix()) {
     a <- NULL
     set <- function(y){
          x<<-y
          a<<- NULL
     }
     get <-function()x
     setInv <- function(inverse) a <<- inverse
     getInv <- function()a
     list(set = set, get = get, setInv = setInv, getInv = getInv)
}


## The cacheSolve function is responsible for inverting the matrix 
##returned by MatrixCacheMatrix and returns that which is the result
##the inverse matrix

cacheSolve <- function(x, ...) {
     ## Return a matrix that is the inverse of 'x'
     a <- x$getInv()
     
     if (!is.null(a)){
          return(a)
     }
     matriz <- x$get()
     a <- solve(matriz, ...)
     x$setInverse(a)           
}
