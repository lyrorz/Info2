import numpy as np
from sklearn.preprocessing import PolynomialFeatures
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
    N,D = xs.shape
    r = np.array([]).reshape(N, -1)
    if degree == 0 or D == 0:
        return np.ones((N, 1))
    for i in range(degree+1):
        r = np.hstack((r, (xs[:, 0].reshape(N, 1)**i)*my_poly_features(xs[:, 1:].reshape(N, -1), degree-i)))
    return r
    pass


def poly_features(xs, degree):
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
    reg = PolynomialFeatures(degree=degree)
    return reg.fit_transform(xs)
x = np.array([[1,2,3,4,5]])
degree = 5
a = my_poly_features(x,degree)
y = poly_features(x,degree)
a = sorted(a.tolist()[0])
y = sorted(y.tolist()[0])
print(a)
print(y)
if a==y:
    print("True")

# TODO: 手动实现