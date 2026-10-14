import torch
from torch import nn, optim

from dataset import load_and_preprocess_data
from model import NIDSModel


def train():
  # Məlumat bazasının (dataset) emalı və yüklənməsi
  train_loader, test_loader, input_dim = load_and_preprocess_data(
      "data/kddcup.data_10_percent.csv"
  )

  # Neyron şəbəkə modelinin və optimizatorun təyin olunması
  model = NIDSModel(input_dim)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  epochs = 5
  print("Modelin öyrədilməsinə başlanılır...")

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} - Loss (Xəta):"
        f" {running_loss/len(train_loader):.4f}"
    )

  # Öyrədilmiş model çəkilərinin yadda saxlanılması
  torch.save(model.state_dict(), "src/nids_model.pth")
  print("Model uğurla öyrədildi və 'src/nids_model.pth' olaraq saxlanıldı!")


if __name__ == "__main__":
  train()