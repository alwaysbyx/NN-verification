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


option = 1;
% input constraint
constraints = [];
x0 = P1(2:1+sizes(1),1);
X0 = P1(2:1+sizes(1),2:1+sizes(1));
ll = max(double(cell2mat(lower(1))),0);
uu = max(double(cell2mat(upper(1))),0);
constraints = [constraints, x0 >= ll, x0 <= uu];
constraints = [constraints, (diag(X0) - (ll+uu).*x0 + uu.*ll <= 1E-5)];

for i = 1:num_hidden_layers
    W_i = double(cell2mat(weights(i)));
    b_i = double(cell2mat(biases(i)));
    Pi = eval(sprintf('P%d',i));
    
    input_span = 2 : 1+sizes(i);
    output_span = 2+sizes(i):1+sizes(i)+sizes(i+1);
    input_linear = Pi(input_span,1);
    output_linear = Pi(output_span, 1);
    output_quadratic = Pi(output_span, output_span);
    cross_terms = Pi(input_span, output_span);
    
    l = double(cell2mat(lower(i+1))); %preactivation
    u = double(cell2mat(upper(i+1))); %preactivation
    ll = max(l,0); %activated
    uu = max(u,0); %activated
    
    % matrix constraints
    constraints = [constraints, Pi(1,1) == 1, Pi >= 0]; 
    % ReLU linear constraints 
    constraints = [constraints, output_linear >= W_i*input_linear + b_i];
    constraints = [constraints, output_linear >=0];
    % ReLU quadratic constraints 
    temp_matrix = W_i*cross_terms;
    constraints = [constraints, diag(output_quadratic) == diag(temp_matrix) + output_linear.*b_i];
    %layer lower/upper constraints
    constraints = [constraints, (diag(output_quadratic) - (ll+uu).*output_linear + ll.*uu <= 1E-5)];
    % consistency constraints, linear
    if option == 1
        if i ~= num_hidden_layers
            nextP = eval(sprintf('P%d',i+1));
            constraints = [constraints, output_linear == nextP(2:1+sizes(i+1),1)];
        end
    else
        if i ~= num_hidden_layers
            nextP = eval(sprintf('P%d',i+1));
            constraints = [constraints, output_linear == nextP(2:1+sizes(i+1),1)];
            constraints = [constraints, Pi(1,output_span) == nextP(1,2:1+sizes(i+1))];
            constraints = [constraints, output_quadratic == nextP(2:1+sizes(i+1),2:1+sizes(i+1))];
        end
    end
    % constraints = [constraints, diag(output_quadratic) - diag(temp_matrix) - b_i.*output_linear - (W_i*input_linear+b_i-output_linear).*ll<=1E-5]; 
    % linear cuts
    constraints = [constraints, (uu-ll).*(W_i*input_linear + b_i-l) + (ll-output_linear).*(u-l) >= 1E-5];
end

s = size(final_linear);
size(final_constant);
dim_final = s(2);
obj = final_linear*output_linear + final_constant;
diagnostics = optimize(constraints, -obj, sdpsettings('dualize', 1, 'solver', 'mosek'))
time = diagnostics.solvertime;
val = value(obj);
save(char(string('SDP_optimum.mat')), 'val', 'time');






