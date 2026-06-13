import torch 
import torch.nn as nn
import torch.nn.functional as F



x = torch.tensor([[1,2,8,9]],dtype=torch.float32)

#############Sigmoid###############
layer= nn.Linear(in_features=4,out_features=2)

output=layer(x)

probabilities= F.sigmoid(output)

print("The probability of the output is : \n",probabilities)

###################Softmax###################

multi_layer=nn.Linear(in_features=4,out_features=3)

output=multi_layer(x)

softmax_probability= F.softmax(output,dim=1)

print("The softmax probability of the output is : \n",softmax_probability)
