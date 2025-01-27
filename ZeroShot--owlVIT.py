import torch
from torch.utils.data import Dataset, DataLoader
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import json
import os
from typing import Dict, Any
import logging
import datetime
import torch.nn.functional as F


# Configure logging with timestamp
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco", 'training.log'))
    ]
)

class ElectricalComponentsDataset(Dataset):
    def __init__(self, json_file: str, image_dir: str, processor: OwlViTProcessor):
        self.image_dir = image_dir
        self.processor = processor
        
        logging.info(f"Loading dataset from {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.categories = data['categories']
        logging.info(f"Found {len(self.categories)} categories")
        
        # Filter out annotations with segmentations
        valid_annotations = [
            ann for ann in data['annotations']
            if not ann.get('segmentation') or len(ann['segmentation']) == 0
        ]
        
        valid_image_ids = set(ann['image_id'] for ann in valid_annotations)
        self.images = [img for img in data['images'] if img['id'] in valid_image_ids]
        self.annotations = valid_annotations
        self.image_to_annotation = {ann['image_id']: ann for ann in self.annotations}
        
        logging.info(f"Loaded {len(self.images)} valid images and {len(self.annotations)} annotations")
        
        # Log category distribution
        category_counts = {}
        for ann in self.annotations:
            cat_id = ann['category_id']
            cat_name = next(cat['name'] for cat in self.categories if cat['id'] == cat_id)
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        logging.info("Category distribution:")
        for cat_name, count in category_counts.items():
            logging.info(f"  {cat_name}: {count} instances")

    def __len__(self) -> int:
        return len(self.images)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_info = self.images[idx]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        annotation = self.image_to_annotation[image_info['id']]
        category = next(cat for cat in self.categories if cat['id'] == annotation['category_id'])
        category_name = category['name']
        bbox = annotation['bbox']
        
        bbox_xyxy = [
            bbox[0],
            bbox[1],
            bbox[0] + bbox[2],
            bbox[1] + bbox[3]
        ]
        
        # Process image and text through processor
        inputs = self.processor(
            images=image,
            text=[category_name],
            return_tensors="pt"
        )
        
        # Create target tensor - ensure single batch dimension
        target = {
            'boxes': torch.tensor(bbox_xyxy, dtype=torch.float).unsqueeze(0),
            'labels': torch.tensor([0], dtype=torch.long),  # using long for labels
            'image_id': torch.tensor([image_info['id']], dtype=torch.long),
            'area': torch.tensor([annotation['area']], dtype=torch.float),
            'iscrowd': torch.tensor([annotation.get('iscrowd', 0)], dtype=torch.long)
        }

        # Remove batch dimension from processor outputs
        return {
            'pixel_values': inputs.pixel_values.squeeze(0),
            'input_ids': inputs.input_ids.squeeze(0),
            'attention_mask': inputs.attention_mask.squeeze(0),
            'target': target  # Keep target as a dictionary with proper tensor shapes
        }
def loss_function(self, outputs, targets):
    # Extract predictions
    pred_logits = outputs.logits[0]  # [num_queries, num_classes]
    pred_boxes = outputs.pred_boxes[0]  # [num_queries, 4]
    
    # Get target boxes and labels
    target_boxes = targets[0]['boxes']  # [num_objects, 4]
    target_labels = targets[0]['labels']  # [num_objects]
    
    # Calculate classification loss
    cls_loss = F.cross_entropy(pred_logits, target_labels)
    
    # Calculate box loss (GIoU)
    giou_loss = torch.zeros(1, device=pred_boxes.device)
    if len(target_boxes) > 0:
        # Calculate GIoU between predictions and targets
        giou = box_giou(pred_boxes, target_boxes)
        giou_loss = (1 - giou).mean()
    
    # Total loss is weighted sum
    total_loss = cls_loss + 2.0 * giou_loss
    return total_loss

def box_giou(boxes1, boxes2):
    """
    Compute GIoU between two sets of boxes
    """
    # Convert to [x1, y1, x2, y2] format if needed
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    
    # Calculate intersection
    inter = box_intersection(boxes1, boxes2)
    
    # Calculate union
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1 + area2 - inter
    
    # Calculate IoU
    iou = inter / (union + 1e-6)
    
    # Calculate enclosing box area
    enclosing_box = torch.zeros_like(boxes1)
    enclosing_box[..., 0] = torch.minimum(boxes1[..., 0], boxes2[..., 0])
    enclosing_box[..., 1] = torch.minimum(boxes1[..., 1], boxes2[..., 1]) 
    enclosing_box[..., 2] = torch.maximum(boxes1[..., 2], boxes2[..., 2])
    enclosing_box[..., 3] = torch.maximum(boxes1[..., 3], boxes2[..., 3])
    enclosing_area = box_area(enclosing_box)
    
    # Calculate GIoU
    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
    return giou

def box_area(boxes):
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])

def box_intersection(boxes1, boxes2):
    x1 = torch.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.minimum(boxes1[..., 3], boxes2[..., 3])
    
    width = (x2 - x1).clamp(min=0)
    height = (y2 - y1).clamp(min=0)
    
    return width * height 
def main():
    # Paths
    json_path = r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco\train\_annotations.coco.json"
    image_dir = r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco\train"
    output_dir = r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco"
    
    # Create timestamp for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = os.path.join(output_dir, f'model_checkpoints_{timestamp}')
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Initialize model and processor
    logging.info("Initializing OwlViT model and processor")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
    model.loss_function = loss_function.__get__(model)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
    
    # Create dataset
    dataset = ElectricalComponentsDataset(
        json_file=json_path,
        image_dir=image_dir,
        processor=processor
    )
    
    # Training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    
    num_epochs = 10
    logging.info(f"Starting training for {num_epochs} epochs")
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Debug info for first batch
            if batch_idx == 0:
                logging.info("First batch structure:")
                for k, v in batch.items():
                    if isinstance(v, dict):
                        logging.info(f"{k}: {type(v)}")
                        for sub_k, sub_v in v.items():
                            logging.info(f"  {sub_k}: {type(sub_v)} - Shape: {sub_v.shape if hasattr(sub_v, 'shape') else 'N/A'}")
                    else:
                        logging.info(f"{k}: {type(v)} - Shape: {v.shape if hasattr(v, 'shape') else 'N/A'}")
            # Move data to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # Ensure target is properly structured
            target = {
                'boxes': batch['target']['boxes'].to(device),
                'labels': batch['target']['labels'].to(device),
                'image_id': batch['target']['image_id'].to(device),
                'area': batch['target']['area'].to(device),
                'iscrowd': batch['target']['iscrowd'].to(device)
            }
            target = [target]  # OwlViT expects a list of targets
            
            # Debug info for first batch in first epoch
            if epoch == 0 and batch_idx == 0:
                logging.info("\nFirst batch model inputs:")
                logging.info(f"pixel_values shape: {pixel_values.shape}")
                logging.info(f"input_ids shape: {input_ids.shape}")
                logging.info(f"attention_mask shape: {attention_mask.shape}")
                logging.info("\nTarget structure:")
                for k, v in target[0].items():
                    logging.info(f"{k}: shape {v.shape}, dtype {v.dtype}")
                
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Calculate loss using the model's loss computation
            loss = model.loss_function(outputs, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        logging.info(f"Epoch {epoch+1} complete - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(model_output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(model_output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, best_model_path)
            logging.info(f"New best model saved with loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()