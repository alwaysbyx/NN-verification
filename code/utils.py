import numpy as np

def create_random_network(input_dimension,
                          num_layers,
                          layer_width,
                          num_classes):
    
    net_layer_types = []
    for i in range(num_layers):
        net_layer_types.append('ff_relu')
    net_layer_types.append('ff')

    net_weights = []
    net_biases = []
    net_weights.append(np.random.rand(layer_width, input_dimension))
    net_biases.append(np.random.rand(layer_width, 1))
    row_dim = layer_width
    for i in range(num_layers - 1):
        net_weights.append(np.random.rand(row_dim+50, row_dim))
        net_biases.append(np.random.rand(row_dim+50, 1))
        row_dim = row_dim+50
    net_weights.append(np.random.rand(num_classes, row_dim))
    net_biases.append(np.random.rand(num_classes, 1))
    return net_weights, net_biases, net_layer_types 
