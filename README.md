# SVD implementation
This project began with me taking a Linear Algebra II. course where I learned about SVD.
After studying it a bit, I wanted to make my own implementation, and I wanted to try using it for image compression with Low-rank matrix approximation.

## Version 0
The first naive version of this algorithm is extremly slow, but it still can compute SVD faster than a human :).
## Version 1
The second version is much faster, and can actually be used for image compression. It uses an iterative algorithm from Wikipedia: https://en.wikipedia.org/wiki/Singular_value_decomposition.
