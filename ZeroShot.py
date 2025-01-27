import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, json_file: str, image_dir: str, transform=None):
        """
        Custom COCO-style dataset for electrical components
        
        Args:
            json_file: Path to the COCO format JSON file
            transform: Optional transforms to apply to images
        """
        self.transform = transform
        
        # Load and parse the JSON data
        with open(json_file, 'r') as f:
            self.coco_data = json.load(f)
            
        # Create category mappings
        self.categories = {cat['id']: cat['name'] 
                         for cat in self.coco_data['categories']}
        
        # Create image id to annotations mapping
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
            
        # Store image information
        self.images = self.coco_data['images']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        img_id = img_info['id']
        
        # Load image
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        annotations = self.image_annotations.get(img_id, [])
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
            
        # Convert to tensor format
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        # Prepare the target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, target

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        num_epochs: int = 100,
        device: str = 'cuda',
        mixed_precision: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} 
                      for t in targets]
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast():
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                self.scaler.scale(losses).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optimizer.step()
            
            total_loss += losses.item()
            
            if batch_idx % 100 == 0:
                print(f'Train Batch [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {losses.item():.6f}')
                
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        
        for images, targets in self.val_loader:
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} 
                      for t in targets]
            
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            
        return total_loss / len(self.val_loader)
    
    def train(self):
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.num_epochs}')
            
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            
            print(f'Train Loss: {train_loss:.6f}')
            print(f'Val Loss: {val_loss:.6f}')
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(self.output_dir, 'best_model.pth'))

def main():
    # Define paths
    json_path = r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco\train\_annotations.coco.json"
    image_dir = r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco\train"
    output_dir = r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up dataset and dataloaders
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = COCODataset(
        json_file=json_path,
        image_dir=image_dir,
        transform=transform
    )
    
    # For validation, we'll use a portion of the training data
    val_dataset = COCODataset(
        json_file=json_path,
        image_dir=image_dir,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Initialize model
    model = OWLViT(num_classes=len(train_dataset.categories))
    
    # Encode class names
    class_names = [train_dataset.categories[i] for i in range(len(train_dataset.categories))]
    model.encode_class_names(class_names)
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mixed_precision=True
    )
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()