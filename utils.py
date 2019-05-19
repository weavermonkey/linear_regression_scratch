import numpy as np

class DataGen:
    def __init__(self,num_samples=100,targ_noise=0.15):
        self.num_samples=num_samples
        self.targ_noise=targ_noise
        self.data = np.random.uniform(low=0.0,high=1.0,size=self.num_samples)
        self.target = (self.data*2) + (np.random.randn(self.data.shape[0]) * targ_noise) #target = data + some variance

class NeuralNet:
    def activation_function(self,input_data, weight):
        return input_data * weight
    
    def cost_function(self,predicted,actual):
        return( (actual - predicted) ** 2 ).sum()
        
    def gradient_descent(self,weight,data,target):
        return data * ( self.activation_function(data,weight) - target) * 2
        
    def update_weights(self, old_weight,data,target,learning_rate):
        return learning_rate * np.mean(self.gradient_descent(old_weight,data,target))
