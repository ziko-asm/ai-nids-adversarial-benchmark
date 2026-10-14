import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from model import NIDSModel

def pgd_attack(model, data, target, epsilon, alpha, num_steps):
    perturbed_data = data.clone().detach()
    perturbed_data = perturbed_data + torch.FloatTensor(*perturbed_data.shape).uniform_(-epsilon, epsilon)
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    for _ in range(num_steps):
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        loss = nn.CrossEntropyLoss()(output, target)
        
        model.zero_grad()
        loss.backward()
        
        data_grad = perturbed_data.grad.data
        perturbed_data = perturbed_data + alpha * data_grad.sign()
        
        eta = torch.clamp(perturbed_data - data, min=-epsilon, max=epsilon)
        perturbed_data = torch.clamp(data + eta, min=0, max=1).detach()
        
    return perturbed_data

def evaluate_robustness_curve():
    print("Robustness Curve analizi başladılır...")
    
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
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    model = NIDSModel(input_dim=input_dim)
    model.load_state_dict(torch.load("nids_robust_model.pth"))
    model.eval()
    
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    robust_accuracies = []
    
    for eps in epsilons:
        if eps == 0.0:
            correct = 0
            total = 0
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
            acc = 100 * correct / total
            robust_accuracies.append(acc)
            print(f"Clean (Epsilon = 0.0) Dəqiqlik: {acc:.2f}%")
            continue
            
        correct_adv = 0
        total = 0
        alpha = eps * 0.25  # Addım ölçüsü
        num_steps = 10
        
        print(f"Epsilon = {eps} üçün PGD hücumu yoxlanılır...")
        for data, target in test_loader:
            adv_data = pgd_attack(model, data, target, eps, alpha, num_steps)
            output_adv = model(adv_data)
            _, predicted_adv = torch.max(output_adv.data, 1)
            correct_adv += (predicted_adv == target).sum().item()
            total += target.size(0)
            
        acc = 100 * correct_adv / total
        robust_accuracies.append(acc)
        print(f"Epsilon {eps} - Robust Accuracy: {acc:.2f}%")
        
    # Qrafikin qurulması və yadda saxlanması
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, robust_accuracies, marker='o', linestyle='-', color='crimson', linewidth=2, markersize=6)
    plt.title('NIDS Robustness Curve under PGD Attack', fontsize=12, fontweight='bold')
    plt.xlabel('Perturbation Magnitude (Epsilon $\epsilon$)', fontsize=10)
    plt.ylabel('Robust Accuracy (%)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 105)
    
    plot_path = "robustness_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nQrafik uğurla yaradıldı və yadda saxlandı: {plot_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_robustness_curve()