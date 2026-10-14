import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Mövcud Baseline MLP Modelimiz"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Conv1DNet(nn.Module):
    """Yeni Əlavə Edilən 1D CNN Modelimiz (Müqayisə üçün)"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.fc1 = nn.Linear(32 * input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # [batch, dim] -> [batch, 1, dim]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    from torch import optim

def train_adversarial_step(model, X_batch, y_batch, epsilon=0.1, alpha=0.01, steps=5):
    """
    Modeli PGD saldırı örnekleriyle eğiterek savunmasını güçlendirir (Adversarial Training).
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    X_adv = X_batch.clone().detach() + torch.FloatTensor(X_batch.shape).uniform_(-epsilon, epsilon)
    X_adv = torch.clamp(X_adv, 0, 1)
    
    for _ in range(steps):
        X_adv.requires_grad = True
        outputs = model(X_adv)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        with torch.no_grad():
            grad = X_adv.grad.sign()
            X_adv = X_adv + alpha * grad
            eta = torch.clamp(X_adv - X_batch, min=-epsilon, max=epsilon)
            X_adv = torch.clamp(X_batch + eta, 0, 1).detach()
            
    optimizer.zero_grad()
    outputs_adv = model(X_adv)
    loss_adv = criterion(outputs_adv, y_batch)
    loss_adv.backward()
    optimizer.step()
    
    return loss_adv.item()

def calculate_feature_importance(model, X_sample):
    """
    Modelin qərar verərkən hansı şəbəkə xüsusiyyətlərinə (features) 
    daha çox önəm verdiyini (XAI / Gradient-based Attribution) hesablayır.
    """
    model.eval()
    X_tensor = X_sample.clone().detach().requires_grad_(True)
    outputs = model(X_tensor)
    
    # Hədəf sinif üzrə gradientləri hesablayırıq
    class_loss = outputs.sum()
    class_loss.backward()
    
    # Gradientlərin mütləq qiyməti xüsusiyyətin əhəmiyyət dərəcəsini göstərir
    importances = torch.abs(X_tensor.grad).mean(dim=0).detach().numpy()
    
    # Normalizasiya edirik (0-100 arası faizə çeviririk)
    if importances.sum() > 0:
        importances = (importances / importances.sum()) * 100
    return importances