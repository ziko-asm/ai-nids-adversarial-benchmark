import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from model import NIDSModel


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def train_adversarial_model():
    print("Adversarial Training başlatılıyor...")
    
    csv_path = "data/kddcup.data_10_percent.csv"
    print(f"Dataset yükleniyor: {csv_path}...")
    df = pd.read_csv(csv_path, header=None)
    
    # Özellikler ve etiketler
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    y_raw = df.iloc[:, -1].values
    y = (y_raw != 'normal.').astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    input_dim = X_scaled.shape[1]
    print(f"Giriş boyutu tespit edildi: {input_dim}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    train_tensor_x = torch.tensor(X_train, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train, dtype=torch.long)
    
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Modeli oluşturuyoruz
    model = NIDSModel(input_dim=input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epsilon = 0.2
    epochs = 3
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data.requires_grad = True
            
            # 1. Temiz veri ile forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # retain_graph=True ekleyerek grafiğin silinmesini önlüyoruz
            loss.backward(retain_graph=True)
            data_grad = data.grad.data
            
            # 2. FGSM ile adversarial veri üretimi
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
            
            # 3. Adversarial veri ile forward pass
            output_adv = model(perturbed_data)
            loss_adv = criterion(output_adv, target)
            
            # Kayıpları birleştirip optimize ediyoruz
            total_loss = (loss + loss_adv) / 2
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
        
    # Modeli kaydediyoruz
    torch.save(model.state_dict(), "nids_robust_model.pth")
    print("Adversarial Training başarıyla tamamlandı! Model 'nids_robust_model.pth' olarak kaydedildi.")

if __name__ == "__main__":
    train_adversarial_model()