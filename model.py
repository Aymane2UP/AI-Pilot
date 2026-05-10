import csv
from neuron import MLP
import json

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

add_size = 60

model = MLP(6,[16,16,2])
len(model.parameters())

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