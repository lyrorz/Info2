import numpy as np
def my_poly_features(xs, degree):
    """Generates polynomial features from given data

    The polynomial features should include monomials (i.e., x_i, x_i**2 etc)
    and interaction terms (x_1*x_2 etc), but no repetitions.
    The order of the samples should not be changed through the transformation.

    Arguments
    xs      2d numpy array of shape (N,D) containing N samples of dimension D
    degree  Maximum degree of polynomials to be considered

    Returns
    An (N,M) numpy array containing the transformed input
    """
    # Your implementation
    pass

x = np.array([[1,2],[5,6]])
y = np.array([[3,4],[7,8]])

# TODO: 手动实现