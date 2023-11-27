Lasso regression, or L1 regularization, is a linear regression technique that incorporates a regularization term in the objective function. The purpose of this regularization term is to encourage simpler models by penalizing the absolute values of the regression coefficients. This leads to sparsity in the model, as some of the coefficients may be exactly zero, effectively performing feature selection.

The Lasso regression objective function is a combination of the standard linear regression objective (mean squared error) and the L1 regularization term. 

Lasso Regression: also known as feature elimination technique(L1 norm) 
A steep slope will lead to overfitting, good for train mode, the test dataset fails
Penalizing higher slopes(m = slope)
The best-fit line has a higher slope, residual error is more, and is overfitting

To prevent that we consider extra parameters(lambda * |slope| in the Cost function Lesser the angle, the lower is the slope, and lower the error, (y - y^)^2
Lambda will always be greater than 0 to any +ve value. Lambda is selected using Cross Validation
Lasso is used for overfitting as well as feature scaling techniques. 

In the case of Ridge Regression, the slope was tending towards zero whereas in Lasso, the slope will actually reach the zero line, and as a result, y = m1x1 + m2x2 + m3x3 + c1(wherever slope is zero, that feature is canceled out)
