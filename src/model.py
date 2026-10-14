from torch import nn


class NIDSModel(nn.Module):

  def __init__(self, input_dim):
    super().__init__()
    # Şəbəkə təbəqələri (Layers)
    self.layer1 = nn.Linear(input_dim, 64)
    self.bn1 = nn.BatchNorm1d(64)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.3)

    self.layer2 = nn.Linear(64, 32)
    self.bn2 = nn.BatchNorm1d(32)

    self.output_layer = nn.Linear(32, 2)  # 2 sinif: Normal (0) və ya Hücum (1)

  def forward(self, x):
    out = self.layer1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.dropout(out)

    out = self.layer2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.output_layer(out)
    return out