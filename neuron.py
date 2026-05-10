from engine import Value
import random

class Neuron:
    def __init__(self,nin):  # nin -> n input
        self.weight = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi * xi for wi , xi in zip(self.weight,x)),self.bias) # (wight * data) + bias
        return act.tanh() # Activation Functions
    
    def parameters(self):
        return self.weight + [self.bias]
    
class Layer:
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [ p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self,nin,nout):
        sz = [nin] + nout
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nout))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [parameter for p in self.layers for parameter in p.parameters()]
        