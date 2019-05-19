import pandas as pd
from utils import NeuralNet,DataGen

#generate dataset with 20 samples, and 'noise' 0.15
ob = DataGen(20,0.15)
ob_nn = NeuralNet()
weights = np.random.rand()

num_iterations = 4
loss = [(weights, ob_nn.cost_function( ob_nn.activation_function(ob.data, weights), ob.target))]
for i in range(num_iterations):
    dw = ob_nn.update_weights(weights, ob.data, ob.target, learning_rate)
    w = w - dw
    loss.append((w, ob_nn.cost_function(ob_nn.activation_function(x, w), t)))
    
for i in range(len(loss)):
    print 'Iteration #' + str(i)
    print '\t value of w0',loss[i][0]
    print '\t current training loss',loss[i][i]
