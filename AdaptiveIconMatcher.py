import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from typing import List, Tuple, Dict

class AdaptiveFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class SymbolEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        # Load pretrained ResNet but remove final layers
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.adaptive_feature = AdaptiveFeatureExtractor(2048)
        
        self.projection = nn.Sequential(
            nn.Conv2d(2048, embedding_dim, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        adapted = self.adaptive_feature(features)
        return self.projection(adapted).squeeze(-1).squeeze(-1)

class SymbolMatcher(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.symbol_encoder = SymbolEncoder(embedding_dim)
        
        # Similarity network for comparing embeddings
        self.similarity_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        query_embedding = self.symbol_encoder(query)
        ref_embedding = self.symbol_encoder(reference)
        
        # Concatenate embeddings for similarity comparison
        combined = torch.cat([query_embedding, ref_embedding], dim=1)
        return self.similarity_net(combined)

class SymbolDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.symbol_pairs = self._load_pairs()
        
    def _load_pairs(self) -> List[Tuple[str, str, int]]:
        """Load positive and negative pairs of symbols"""
        pairs = []
        symbols = os.listdir(self.data_dir)
        
        # Create positive pairs (same symbol, different styles)
        for symbol in symbols:
            symbol_dir = os.path.join(self.data_dir, symbol)
            if not os.path.isdir(symbol_dir):
                continue
                
            files = os.listdir(symbol_dir)
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    pairs.append((
                        os.path.join(symbol_dir, files[i]),
                        os.path.join(symbol_dir, files[j]),
                        1
                    ))
        
        # Create negative pairs (different symbols)
        num_pos = len(pairs)
        symbol_list = list(symbols)
        for _ in range(num_pos):
            sym1, sym2 = np.random.choice(symbol_list, 2, replace=False)
            files1 = os.listdir(os.path.join(self.data_dir, sym1))
            files2 = os.listdir(os.path.join(self.data_dir, sym2))
            
            pairs.append((
                os.path.join(self.data_dir, sym1, np.random.choice(files1)),
                os.path.join(self.data_dir, sym2, np.random.choice(files2)),
                0
            ))
            
        return pairs
    
    def __len__(self) -> int:
        return len(self.symbol_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img1_path, img2_path, label = self.symbol_pairs[idx]
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, label

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
) -> Dict[str, List[float]]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (query, reference, labels) in enumerate(train_loader):
            query, reference = query.to(device), reference.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(query, reference).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for query, reference, labels in val_loader:
                query, reference = query.to(device), reference.to(device)
                labels = labels.float().to(device)
                
                outputs = model(query, reference).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.4f}\n')
        
    return history

def predict_symbols(
    model: nn.Module,
    floor_plan: torch.Tensor,
    reference_symbols: Dict[str, torch.Tensor],
    confidence_threshold: float = 0.7,
    device: str = 'cuda'
) -> Dict[str, List[Tuple[int, int, float]]]:
    """
    Detect symbols in floor plan using reference symbols from legend
    
    Args:
        model: Trained SymbolMatcher model
        floor_plan: Tensor of floor plan image
        reference_symbols: Dict mapping symbol names to their tensor representations
        confidence_threshold: Minimum confidence score for detection
        device: Device to run inference on
        
    Returns:
        Dictionary mapping symbol names to lists of (x, y, confidence) detections
    """
    model = model.to(device)
    model.eval()
    
    detections = {}
    
    with torch.no_grad():
        for symbol_name, reference in reference_symbols.items():
            reference = reference.to(device)
            floor_plan = floor_plan.to(device)
            
            # Get similarity scores
            scores = model(floor_plan, reference)
            
            # Get locations where confidence exceeds threshold
            locations = torch.where(scores > confidence_threshold)
            
            detections[symbol_name] = [
                (int(x), int(y), float(scores[x, y]))
                for x, y in zip(*locations)
            ]
            
    return detections

# Example usage:
if __name__ == "__main__":
    # Initialize model and datasets
    model = SymbolMatcher()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = SymbolDataset("path/to/train/data", transform=transform)
    val_dataset = SymbolDataset("path/to/val/data", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Save trained model
    torch.save(model.state_dict(), "symbol_matcher.pth")