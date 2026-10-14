import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, header=None)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Bütün sütunları tək-tək yoxlayıb rəqəmə çeviririk
    X_processed = []
    for i in range(X.shape[1]):
        col = X[:, i]
        if col.dtype == object or isinstance(col[0], str):
            le = LabelEncoder()
            col = le.fit_transform(col.astype(str))
        X_processed.append(col.astype(np.float32))
        
    X = np.column_stack(X_processed)
    
    y = [0 if str(label).strip() == 'normal.' else 1 for label in y]
    y = torch.tensor(y, dtype=torch.long)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader, X.shape[1]