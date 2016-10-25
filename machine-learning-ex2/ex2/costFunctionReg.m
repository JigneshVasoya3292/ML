function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hypothesis = sigmoid(X * theta);

J = sum(y.*log(hypothesis) + ((1 -y).*log(1 - hypothesis)));
J = (-1/m) * J;
reg_term = 0;

for  i = 2 : size(theta)
	reg_term = reg_term + (theta(i) ^ 2 );
end

reg_term = reg_term * (lambda/ (2*m));

J = J + reg_term;

for i = 1:length(theta)
    sigma = 0;
	reg_term;
   	sigma =  sum((hypothesis - y).*X(:, i));
   	sigma = (1/m) * sigma;

   	% add reg_term;
   	if i >= 2 
   		reg_term = (lambda/m) * theta(i);
   	else
   		% in case of theta(0)
   		reg_term = 0;
   	end
   	grad(i) = sigma + reg_term;
end


% =============================================================

end
