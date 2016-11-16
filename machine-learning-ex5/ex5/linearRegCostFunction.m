function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hypothesis = X * theta;

J = sum((hypothesis - y) .^ 2);

J = J * (1/(2*m));

reg_term = (lambda/(2*m)) * sum(sum(theta(2:end) .^ 2));
 
J = J + reg_term;

reg_theta = theta;
reg_theta(1) = 0;

grad = (1/m) * (X' * (hypothesis - y));
grad_reg = (lambda/m) * reg_theta;
grad = grad + grad_reg;




% =========================================================================

grad = grad(:);

end
