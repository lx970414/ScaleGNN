import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

from model import ScaleGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid(root='/tmp/Cora', name='Cora') # replecement
data = dataset[0].to(device)


in_channels = dataset.num_features
hidden_channels = 128  
out_channels = dataset.num_classes
num_layers = 3 
dropout = 0.5  
lr = 0.01 
epochs = 200 

model = ScaleGNN(in_channels, hidden_channels, out_channels, num_layers, dropout).to(device)

optimizer = Adam(model.parameters(), lr=lr)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # 前向传播
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])  # 计算损失
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
    accuracy = correct / data.test_mask.sum().item()
    return accuracy

best_val_acc = 0
for epoch in range(epochs):
    loss = train()
    val_acc = test()

    print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_scaleGNN_model.pth')

print('Training complete.')
