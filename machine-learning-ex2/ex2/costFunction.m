function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

hypothesis = sigmoid(X * theta);

%for i = 1:m
%	J = J + (y(i) * log(sigmoid(X(i, :) * theta)) + ((1 - y(i)) * (log((1 - sigmoid(X(i, :) * theta))))));
%end

%for i = 1:m
%	J = J + (y(i) * log(hypothesis(i)) + ((1 - y(i)) * (log((1 - hypothesis(i))))));
%end

J = sum(y.*log(hypothesis) + ((1 -y).*log(1 - hypothesis)));
J = (-1/m) * J;


for i = 1:length(theta)
    sigma = 0;

    %for j = 1:m
    %   sigma = sigma + ((hypothesis(i)- y(i)) * X(j, i));
   	%end

   	sigma =  sum((hypothesis - y).*X(:, i));
   	grad(i) = (1/m) * sigma;
end



% =============================================================

end
