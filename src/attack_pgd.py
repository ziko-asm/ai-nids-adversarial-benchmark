import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from model import NIDSModel

def pgd_attack(model, data, target, epsilon, alpha, num_steps):
    # Orijinal datanın surətini çıxarıb üzərində dəyişiklik edirik
    perturbed_data = data.clone().detach()
    
    # Random start (hücumun effektivliyini artırmaq üçün təsadüfi başlanğıc)
    perturbed_data = perturbed_data + torch.FloatTensor(*perturbed_data.shape).uniform_(-epsilon, epsilon)
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    for _ in range(num_steps):
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        loss = nn.CrossEntropyLoss()(output, target)
        
        # Gradientləri hesablayırıq
        model.zero_grad()
        loss.backward()
        
        # PGD addımı (Projected Gradient Descent)
        data_grad = perturbed_data.grad.data
        perturbed_data = perturbed_data + alpha * data_grad.sign()
        
        # Epsilon sərhədləri daxilində saxlayırıq (Clipping)
        eta = torch.clamp(perturbed_data - data, min=-epsilon, max=epsilon)
        perturbed_data = torch.clamp(data + eta, min=0, max=1).detach()
        
    return perturbed_data

def evaluate_pgd_defense():
    print("PGD Attack Evaluation başladı...")
    
    csv_path = "data/kddcup.data_10_percent.csv"
    df = pd.read_csv(csv_path, header=None)
    
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    y_raw = df.iloc[:, -1].values
    y = (y_raw != 'normal.').astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    input_dim = X_scaled.shape[1]
    
    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    test_tensor_x = torch.tensor(X_test, dtype=torch.float32)
    test_tensor_y = torch.tensor(y_test, dtype=torch.long)
    
    test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Gücləndirilmiş modelimizi yükləyirik
    model = NIDSModel(input_dim=input_dim)
    model.load_state_dict(torch.load("nids_robust_model.pth"))
    model.eval()
    
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    epsilon = 0.2
    alpha = 0.05
    num_steps = 10
    
    print("Test paketləri PGD hücumu ilə sınanır...")
    for data, target in test_loader:
        # 1. Təmiz data dəqiqliyi
        output_clean = model(data)
        _, predicted_clean = torch.max(output_clean.data, 1)
        correct_clean += (predicted_clean == target).sum().item()
        
        # 2. PGD hücumuna məruz qalmış adversarial data
        adv_data = pgd_attack(model, data, target, epsilon, alpha, num_steps)
        output_adv = model(adv_data)
        _, predicted_adv = torch.max(output_adv.data, 1)
        correct_adv += (predicted_adv == target).sum().item()
        
        total += target.size(0)
        
    clean_acc = 100 * correct_clean / total
    adv_acc = 100 * correct_adv / total
    
    print(f"\n--- PGD BENCKMARK NƏTİCƏLƏRİ ---")
    print(f"Təmiz Data Dəqiqliyi (Clean Accuracy): {clean_acc:.2f}%")
    print(f"PGD Hücumu Altında Dəqiqlik (Robust Accuracy): {adv_acc:.2f}%")

if __name__ == "__main__":
    evaluate_pgd_defense()