from dataset import load_and_preprocess_data
from model import NIDSModel
import torch
import torch.nn as nn


def fgsm_attack(model, criterion, inputs, labels, epsilon):
  # Giriş verilənlərinin gradientini tələb edirik
  inputs.requires_grad = True

  # Modeldən keçirib proqnoz alırıq
  outputs = model(inputs)
  loss = criterion(outputs, labels)

  # Modelin çəkilərinə görə gradientləri sıfırlayıb geri yayılma (backprop) edirik
  model.zero_grad()
  loss.backward()

  # Giriş xüsusiyyətlərinin işarəsini (sign) götürürük və hücum tətbiq edirik
  data_grad = inputs.grad.data
  perturbed_inputs = inputs + epsilon * data_grad.sign()

  return perturbed_inputs


def evaluate_adversarial_robustness():
  # Məlumatları yükləyirik
  _, test_loader, input_dim = load_and_preprocess_data(
      "data/kddcup.data_10_percent.csv"
  )

  # Öyrədilmiş modelimizi yükləyirik
  model = NIDSModel(input_dim)
  model.load_state_dict(torch.load("src/nids_model.pth"))
  model.eval()

  criterion = nn.CrossEntropyLoss()

  epsilon = 0.4  # Hücum şiddəti
  correct = 0
  adversarial_correct = 0
  total = 0

  print(f"FGSM Hücumu test edilir (Epsilon = {epsilon})...")

  for inputs, labels in test_loader:
    # 1. Təmiz (normal) məlumatla dəqiqlik
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    # 2. Hücuma məruz qalmış (adversarial) məlumatla dəqiqlik
    perturbed_inputs = fgsm_attack(model, criterion, inputs, labels, epsilon)
    adv_outputs = model(perturbed_inputs)
    _, adv_predicted = torch.max(adv_outputs.data, 1)
    adversarial_correct += (adv_predicted == labels).sum().item()

  clean_accuracy = 100 * correct / total
  adv_accuracy = 100 * adversarial_correct / total

  print(f"\n--- NƏTİCƏLƏR ---")
  print(f"Təmiz Modelin Dəqiqliyi (Clean Accuracy): {clean_accuracy:.2f}%")
  print(
      f"Hücumdan Sonrakı Dəqiqlik (Adversarial Accuracy):"
      f" {adv_accuracy:.2f}%"
  )


if __name__ == "__main__":
  evaluate_adversarial_robustness()