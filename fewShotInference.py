import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import tempfile
from pathlib import Path
import torch.nn.functional as F
from fewShot import create_symbol_matcher

class FloorPlanProcessor:
    def __init__(self, model_path, device='cuda', window_size=224, stride=112):
        self.device = device
        self.window_size = window_size
        self.stride = stride
        
        # Load model
        self.model = create_symbol_matcher().to(device)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((window_size, window_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_windows(self, image):
        """Extract overlapping windows from image."""
        width, height = image.size
        windows = []
        positions = []
        
        for y in range(0, height - self.window_size + 1, self.stride):
            for x in range(0, width - self.window_size + 1, self.stride):
                window = image.crop((x, y, x + self.window_size, y + self.window_size))
                windows.append(self.transform(window))  # Image is already RGB
                positions.append((x, y))
                
        return torch.stack(windows), positions

    def non_max_suppression(self, boxes, scores, threshold=0.3):
        """Apply NMS to remove overlapping detections."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(0, descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
                
            i = order[0]
            keep.append(i.item())
            
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= threshold).nonzero().squeeze()
            
            if ids.numel() == 0:
                break
            order = order[ids + 1]
            
        return torch.tensor(keep)

    def process_floorplan(self, floorplan_path, reference_symbols, confidence_threshold=0.5, nms_threshold=0.3):
        """
        Process a floor plan and detect symbols.
        
        Args:
            floorplan_path: Path to floor plan image
            reference_symbols: List of (symbol_image, symbol_name) tuples
            confidence_threshold: Minimum confidence for detection
            nms_threshold: NMS IoU threshold
        
        Returns:
            List of (symbol_name, box, confidence) tuples
        """
        # Load floor plan and convert to RGB
        floorplan = Image.open(floorplan_path).convert('RGB')
        windows, positions = self.extract_windows(floorplan)
        
        all_detections = []
        
        # Process each reference symbol
        for ref_image, symbol_name in reference_symbols:
            # Ensure reference image is RGB
            ref_image_rgb = ref_image.convert('RGB')
            ref_tensor = self.transform(ref_image_rgb).unsqueeze(0).to(self.device)
            
            # Process in batches
            batch_size = 32
            all_confidences = []

            for i in range(0, len(windows), batch_size):
                batch = windows[i:i + batch_size].to(self.device)
                with torch.no_grad():
                    similarity = self.model(ref_tensor, batch)
                    confidence = torch.sigmoid(similarity).squeeze()  # Add squeeze()
                    all_confidences.append(confidence.cpu())  # Keep as tensor

            # Convert to numpy at the end
            confidences = torch.cat(all_confidences).numpy()  # Use torch.cat instead of extending list
            
            # Filter by confidence
            high_conf_idx = np.where(confidences > confidence_threshold)[0]
            
            if len(high_conf_idx) > 0:
                # Create boxes for NMS
                boxes = []
                for idx in high_conf_idx:
                    x, y = positions[idx]
                    boxes.append([x, y, x + self.window_size, y + self.window_size])
                
                boxes = torch.tensor(boxes).float()
                scores = torch.tensor(confidences[high_conf_idx]).float()
                
                # Apply NMS
                keep_idx = self.non_max_suppression(boxes, scores, nms_threshold)
                
                # Store detections
                for idx in keep_idx:
                    box = boxes[idx].tolist()
                    conf = float(scores[idx])
                    all_detections.append((symbol_name, box, conf))
        
        return all_detections

def visualize_detections(image_path, detections, output_path=None):
    """Visualize detections on the image."""
    import cv2
    
    # Load image
    # Load with PIL and convert to RGB first
    image = Image.open(str(image_path)).convert('RGB')
    # Convert to OpenCV format for drawing
    image = np.array(image)
    image = image[:, :, ::-1].copy()  # RGB to BGR for OpenCV
    
    # Draw detections
    for symbol_name, box, conf in detections:
        x1, y1, x2, y2 = [int(c) for c in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{symbol_name}: {conf:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(str(output_path), image)
    return image

# Usage example:
if __name__ == "__main__":
    model_path = r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco\best_model.pth"
    processor = FloorPlanProcessor(model_path)
    
    # Load reference symbols
    reference_symbols = [
        (Image.open("icon.png"), "ress_led_ceiling_light"),
        (Image.open("icon_switch.png"), "Switch"),
        # Add more symbols as needed
    ]
    
    # Process floor plan
    detections = processor.process_floorplan(
        "656 Townsend.png",
        reference_symbols,
        confidence_threshold=0.7,
        nms_threshold=0.3
    )
    
    # Visualize results
    visualize_detections("656 Townsend.png", detections, "output.png")
    
    # Print results
    for symbol_name, box, conf in detections:
        print(f"Found {symbol_name} at {box} with confidence {conf:.2f}")