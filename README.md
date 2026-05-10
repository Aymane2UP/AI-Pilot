### Project Name : **AI Pilot** model

- First let's read our [*CSV*](SmartRocketTrainingData20260507_191122.csv) data file 
We don't use Pandas or Pytorch because we want to understand the under the hood concepts of trining the models this the bigger idea we work for here


```python
import csv

file_path = "SmartRocketTrainingData20260507_191122.csv"
data = []

with open(file_path,'r') as file:
    reader = csv.reader(file)
    header = next(reader)

    for row in reader:
        try:
            presed_row_value = [float(row_value) for row_value in row]
            data.append(presed_row_value)
        except ValueError:
            continue

```

Now let's filter the Data in inputs X and outputs Y


```python
X,Y = [],[]
for row in data:
    if row[2] == -1: continue

    inputs = [
        row[3], # DistanceX
        row[4], # DistanceY
        row[7], # VelocityX
        row[8], # VelocityY
        row[9], # Angle
        row[10] # AngularVelocity
    ]
    inputs[0] = inputs[0] / 60.0  # div by the max for all
    inputs[1] = inputs[1] / 60.0
    inputs[2] = inputs[2] / 15.0
    inputs[3] = inputs[3] / 15.0
    inputs[4] = inputs[4] / 180
    inputs[5] = inputs[5] / 30

    labels = [
        row[0], # Speed -> Engine thrust
        row[1]  # Steering -> rocket direction
    ]

    labels[0] = labels[0] / 30.0

    X.append(inputs)
    Y.append(labels)

```


```python
from engine import Value
import random
```


```python

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
        
```

### Sanity Check


```python
xs = [
  [0.2, 0.5, -0.1, 0.8, 0.0, -0.5], 
  [0.9, -0.9, 0.5, 0.1, 0.2, 0.8]   
]

ys = [
  [1.0, 0.0],  
  [-1.0, 1.0]  
]
model = MLP(6,[16,16,2])
for x in xs:
  model(x)
  
len(model.parameters())
```




    418




```python
LR = -0.01 # Learning Rate
for iLoop in range(10000):
    ypred = [model(x) for x in xs] 
    loss = sum((yout - yget)**2 for y1 , y2 in zip(ys,ypred) for yget , yout in zip(y1,y2))
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()
    for p in model.parameters():
        p.data += LR * p.grad
    print(iLoop,"  ",loss.data)
```

    0    0.26337698776060003
    1    0.24631440356935275
    2    0.2296307883706697
    3    0.21340214847545352
    ...   ............
    9997    3.7091882477206207e-06
    9998    3.709139458770635e-06
    9999    3.7090906719262012e-06
    


```python
[[yy.data for yy in y] for y in ypred]

```




    [[0.9986639507439349, -1.3248433125005435e-06],
     [-0.9999930011368631, 0.998612912289003]]



### Save Result as [CSV](Ai-Pilot_SaveData.csv) file


```python
save_path = "Ai-Pilot_SaveData.csv"

with open(save_path,"w",newline='') as save_csv:
    writer = csv.writer(save_csv,delimiter=',')
    data = [[yy.data for yy in y] for y in ypred]
    for dataLine in data:
        print(dataLine)
        writer.writerow(dataLine)
```

    [-0.0007532713450463511, 0.001072737118095538]
    [-0.0008018101785604881, 0.0010532406912863485]
    [-0.0008592046996735258, 0.0010729882572905875]
    .......................
    [-0.0009413359736153931, 0.001114161214091576]
    [-0.0009413359614796558, 0.0011141611509858016]
    [-0.0009413359663976974, 0.0011141611498034993]
    

### Now The really training


```python
add_size = 60

model = MLP(6,[16,16,2])
len(model.parameters())
```




    418




```python
for i in range(0,len(X),60):
    x = []
    y = []
    for i in range(i,i+add_size):
        x.append(X[i])
        y.append(Y[i])
 


    LR = -0.0001 # Learning Rate
    for iLoop in range(60): 
        ypred = [model(xi) for xi in x] 
        loss = sum((yout - yget)**2 for y1 , y2 in zip(y,ypred) for yget , yout in zip(y1,y2))
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        for p in model.parameters():
            p.data += LR * p.grad
        print(iLoop,"  ",loss.data)
```


```python
model.layers
```




    [<__main__.Layer at 0x1b8c99a8750>,
     <__main__.Layer at 0x1b82987e5d0>,
     <__main__.Layer at 0x1b82987f4d0>]




```python
import json
```


```python
save_wight_path = "Ai-Piloat_Weights.json"
def export_brain(model):
    print("--- Copy weights to Json file ---")
    model_brain = []
    for i ,layer in enumerate(model.layers):
        for j , neuron in enumerate(layer.neurons):
            layer_data = {
                "weights" : [w.data for w in layer.neurons[j].weight],
                "bias" : neuron.bias.data
            }
            print([w.data for w in layer.neurons[j].weight])
            model_brain.append(layer_data)
    with open(save_wight_path,"w") as save_json:
        json.dump(model_brain,save_json)
    print(len(model_brain))
export_brain(model)
```

    --- Copy weights to Json file ---
    [0.07535985450426193, -0.49974271057015907, 0.19048762336059716, 0.5436205839111302, -0.14800972374594973, -0.8077646964994691]
    [-0.508903037298761, 0.18650196377952635, -0.6948947535074838, -0.4416792738827003, 0.102114132739158, -0.23264652601602576]
    [-0.8930622037218395, 0.4044740766639448, 0.43018185677985543, 0.8312450327186477, -0.8220908611221746, -0.8471233218328145]
    ...........
    [-0.7585745604246629, 0.08722464148084846, 0.6424633944880712, 0.299192938379236, 0.06918632680229403, -0.07507428360653465, -0.37226258709243, -0.27308075296895423, 0.5970942134285295, -0.12811546097083318, -0.20594090233388002, -0.729497712778941, -0.2891786760185148, 0.476480677026851, -0.8559440475344677, -0.12977119411062374]
    34
    
