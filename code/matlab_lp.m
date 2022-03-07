function [val] = matlab_lp(matlab_folder)

cd(matlab_folder)
addpath(genpath('/Users/bb/Desktop/UCSD/2022Winter/ECE285/yalmip/YALMIP-master'))
addpath(genpath('/Users/bb/mosek/mosek/9.3/toolbox/r2015a'))

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

% Variable
size_neuron = sum(sizes);
x = sdpvar(size_neuron,1);
x0 = x(1:sizes(1));
current_pos = 1;

% input constraint
constraints = [x0>=double(cell2mat(lower(1))), x0<=double(cell2mat(upper(1)))];

for i = 1:num_hidden_layers
    W_i = double(cell2mat(weights(i)));
    b_i = double(cell2mat(biases(i)));
    x_input = x(current_pos: current_pos + sizes(i) - 1);
    x_output = x(current_pos + sizes(i): current_pos + sizes(i) + sizes(i+1) - 1);
    
    % ReLU linear constraints 
    constraints = [constraints, x_output >= W_i*x_input + b_i];
    constraints = [constraints, x_output >=0];
    % ReLU convex relaxation
    ll = double(cell2mat(lower(i+1)));
    uu = double(cell2mat(upper(i+1)));
    constraints = [constraints, (max(uu,0)-max(ll,0)).*(W_i*x_input + b_i-ll) + (max(ll,0)-x_output).*(uu-ll)>=0];
    % ki = (max(0,uu)-max(0,ll))./(uu-ll);
    % constraints = [constraints, x_output <= ki.*(W_i*x_input + b_i-ll) + max(0,ll) + 1e-5];
    constraints = [constraints, W_i*x_input + b_i <= uu, W_i*x_input + b_i >= ll];
    
    current_pos = current_pos + size(i);
end


% Objective
s = size(final_linear);
size(final_constant);
dim_final = s(2);
obj = final_linear*x_output + final_constant;
diagnostics = optimize(constraints, -obj, sdpsettings('dualize', 1,'solver', 'mosek'))
time = diagnostics.solvertime;
if diagnostics.problem == 0
    val = value(obj);
else
    val = diagnostics.problem;
end
save(char(string('SDP_optimum.mat')), 'val', 'time');
exit



