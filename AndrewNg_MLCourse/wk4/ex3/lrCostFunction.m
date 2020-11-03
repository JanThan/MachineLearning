function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

z = X*theta;
g = sigmoid(z);
h=g;
% 
% % SUM1 = -y' * log(h);
% % SUM2 = (1-y)'*log(1-h);
SUM = (-y' * log(h)) - ((1-y)'*log(1-h));
REG = sum(theta(2:end).^2);
J = 1/m * SUM + lambda/(2*m) * REG;

SUM = ((h-y)'*X(:,1));
grad(1) = 1/m * SUM;
SUM = ((h-y)'*X(:,2:end))';
REG = theta(2:end);
grad(2:size(theta)) = 1/m * SUM + lambda/m * REG ;


%% Both Methods work, above is prefered due to full vectorisation
% SUM = sum((h - y)' * X);
% REG = theta(2:end);
% grad = 1/m * SUM + lambda/m * REG;

% z = X*theta;
% g = sigmoid(z);
% h = g;
% SUM = sum(-y.*log(h) - (1-y).*log(1-h));
% REG = sum(theta(2:end).^2);
% J = 1/m * SUM + lambda/(2*m) * REG;
% 
% grad(1) = 1/m .* ((h-y)'*X(:,1));
% grad(2:size(theta)) = (1/m .* ((h-y)'*X(:,2:end)))' + lambda/m * theta(2:end);

% =============================================================

grad = grad(:);

end
