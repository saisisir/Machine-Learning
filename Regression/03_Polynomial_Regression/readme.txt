Polynomial regression is a type of linear regression that models the relationship between independent and dependent variables as an nth-degree polynomial. The goal is to find the best way to draw a line using data points. 

f ( x ) = c0 + c1 x + c2 x2 ⋯ cn xn where n is the degree of the polynomial and c is a set of coefficients.   

Polynomial regression models are usually fit with the method of least squares. The \(R\)-squared value determines how good the fitting is. For example, an \(R\)-squared value of 0.735 indicates a good fitting but not very strong.   

One possible strategy for choosing the order of an approximate polynomial is to successively fit the models in increasing order and test the significance of regression coefficients at each step of model fitting.

The goal of polynomial regression is to model a non-linear relationship between the independent and dependent variables (technically, between the independent variable and the conditional mean of the dependent variable).
