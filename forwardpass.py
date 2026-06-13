import torch 

x= torch.tensor([[1,3,6,8]])

w= torch.tensor([[1,0.5,0.7,0.3],[1,5,9,3]])

bias=torch.tensor([0.5,1])

def forward(x,w,bias):
    forward_pass= torch.matmul(x,w.T)+bias
    return forward_pass

output=forward(x,w,bias)
print(output)