function [val] = matlab_sdp3(matlab_folder)

% Reading the mat files 
S = load('sizes.mat');
sizes = double(S.sizes);
B = load('biases.mat');
biases = squeeze(B.biases).';
W = load('weights.mat');
weights = squeeze(W.weights);
O = load('opt_params.mat');
opt_params = O.opt_params;
lower = [];
lower = [squeeze(opt_params.lower)];
upper = [];
upper = [squeeze(opt_params.upper)];
final_constant = double(opt_params.final_constant);
final_linear = double(opt_params.final_linear);

% Computing number of hidden layers 
s = size(weights);
num_hidden_layers = s(2);
if(s(1) ~= 1)
  weights = {};
  weights{1} = squeeze(W.weights);
  biases = {};
  biases{1} = (squeeze(B.biases)).';
  num_hidden_layers = 1;
end 

for i = 1:size(sizes,2)-1
    size_layer_i = 1 + sizes(i) + sizes(i+1);
    str = sprintf('sdpvar(%d,%d)', size_layer_i, size_layer_i);
    ci = sprintf('P%d=%s;', i, str);
    eval(ci);
end

% input constraint
constraints = [];
x0 = P1(2:1+sizes(1),1);
X0 = P1(2:1+sizes(1),2:1+sizes(1));
ll = double(cell2mat(lower(1)));
uu = double(cell2mat(upper(1)));
constraints = [constraints, x0 >= ll, x0 <= uu];
constraints = [constraints, (diag(X) - (ll+uu).*x + uu.*ll <= 1E-5)];

for i = 1:num_hidden_layers
    W_i = double(cell2mat(weights(i)));
    b_i = double(cell2mat(biases(i)));
    Pi = eval(sprintf('P%d',i));
    
    in_span = 2 : 1+sizes(i);
    out_span = 2+sizes(i):1+sizes(i)+sizes(i+1);
    input_linear = Pi(input_span,1);
    output_linear = Pi(output_span, 1);
    output_quadratic = Pi(output_span, output_span);
    cross_terms = Pi(input_span, output_span);
    
    % matrix constraints
    constraints = [constraints, Pi(1,1) == 1, Pi >= 0]; 
    % ReLU linear constraints 
    constraints = [constraints, output_linear >= W_i*input_linear + b_i];
    constraints = [constraints, output_linear >=0];
    % ReLU quadratic constraints 
    temp_matrix = W_i*cross_terms;
    constraints = [constraints, diag(output_quadratic) == diag(temp_matrix) + output_linear.*b_i];
    % consistency constraints, linear
    if i ~= num_hidden_layers
        nextP = eval(sprintf('P%d',i+1));
        constraints = [constraints, Pi(outspan) == nextP(2:1+sizes(i+1),1)];
    end
    % linear cuts
    
end



