function [val] = matlab_sdp(matlab_folder)

cd(matlab_folder)
addpath(genpath('/Users/bb/Desktop/UCSD/2022Winter/ECE285/yalmip/YALMIP-master'))
addpath(genpath('/Users/bb/mosek/mosek/9.3/toolbox/r2015a'))

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


size_big_matrix = 1 + sum(sizes);

M = sdpvar(size_big_matrix, size_big_matrix);
constraints = [M>=0, M(1, 1) == 1];
x = M(1, 2: 1+sizes(1)).';
X = M(2: 1 + sizes(1), 2: 1 + sizes(1));

%Input constraints 
constraints = [constraints, x>=double(cell2mat(lower(1)))];
constraints = [constraints, x<=double(cell2mat(upper(1)))];
constraints = [constraints, (diag(X) - (double(cell2mat(lower(1))) + double(cell2mat(upper(1)))).*x ...
+ double(cell2mat(lower(1))).*double(cell2mat(upper(1))) <= 1E-5)];

current_pos_matrix = 1;

for i = 1:num_hidden_layers
    W_i = double(cell2mat(weights(i)));
    b_i = double(cell2mat(biases(i)));
    input_span = 1 + current_pos_matrix: current_pos_matrix + sizes(i);
    output_span = 1 + current_pos_matrix + sizes(i): current_pos_matrix + sizes(i) + sizes(i+1);
    input_linear = M(1, input_span).';
    output_linear = M(1, output_span).';
    output_quadratic = M(output_span, output_span);
    cross_terms = M(input_span, output_span);
    
    % ReLU linear constraints 
    constraints = [constraints, output_linear >= W_i*input_linear + b_i];
    constraints = [constraints, output_linear >=0];
    % ReLU quadratic constraints 
    temp_matrix = W_i*cross_terms;
    constraints = [constraints, diag(output_quadratic) == diag(temp_matrix) + output_linear.*b_i];
    
    % layerwise constraints 
    constraints = [constraints, (diag(output_quadratic) - (double(cell2mat(lower(i+1))) + double(cell2mat(upper(i+1)))).*output_linear ...
				 + double(cell2mat(lower(i+1))).*double(cell2mat(upper(i+1))) <= 1E-5)];
    
    current_pos_matrix = current_pos_matrix + sizes(i);

    % New constraint 1 
    constraints = [constraints, diag(output_quadratic) - diag(temp_matrix) - b_i.*output_linear - double(cell2mat(lower(i+1))).*output_linear + (W_i*input_linear).*double(cell2mat(lower(i+1))) + double(cell2mat(lower(i+1))).*b_i<=1E-5]; 
    
end

% Constructing the objective 
tic;
s = size(final_linear);
size(final_constant);
dim_final = s(2);
y_final = M(1, 1 + current_pos_matrix: current_pos_matrix + dim_final).';
obj = final_linear*y_final + final_constant;
diagnostics = optimize(constraints, -obj, sdpsettings('dualize', 1, 'solver', 'mosek'))
time = diagnostics.solvertime;
val = value(obj);
save(char(string('SDP_optimum.mat')), 'val', 'time');
exit
