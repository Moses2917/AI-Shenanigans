import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class SymbolDataset(Dataset):
    def __init__(self, json_path, image_dir, image_size=800):
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
            
        self.image_dir = image_dir
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        # Group annotations by category
        self.category_to_images = {}
        for ann in self.annotations:
            cat_id = ann['category_id']
            if cat_id not in self.category_to_images:
                self.category_to_images[cat_id] = []
            self.category_to_images[cat_id].append(
                (ann['image_id'], ann['bbox'])
            )
            
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, image_id, bbox):
        image_info = next(img for img in self.images if img['id'] == image_id)
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Crop to bbox
        x, y, w, h = [int(c) for c in bbox]
        image = image.crop((x, y, x+w, y+h))
        return self.transform(image)

    def __getitem__(self, idx):
        # Select random category
        cat_id = random.choice(list(self.category_to_images.keys()))
        images = self.category_to_images[cat_id]
        
        # Select reference and positive example
        ref_idx, pos_idx = random.sample(range(len(images)), k=2)
        reference = self.load_image(*images[ref_idx])
        positive = self.load_image(*images[pos_idx])
        
        # Select negative examples from other categories
        other_cats = list(set(self.category_to_images.keys()) - {cat_id})
        neg_images = []
        for _ in range(3):  # 3 negative examples
            neg_cat = random.choice(other_cats)
            neg_sample = random.choice(self.category_to_images[neg_cat])
            neg_images.append(self.load_image(*neg_sample))
            
        candidates = torch.stack([positive] + neg_images)
        positive_idx = torch.tensor([0])  # First candidate is positive
        
        return {
            'reference': reference,
            'candidates': candidates,
            'positive_idx': positive_idx,
            'category': self.categories[cat_id]
        }

    def __len__(self):
        return sum(len(imgs) for imgs in self.category_to_images.values())

def get_dataloaders(json_path, image_dir, batch_size=28):
    dataset = SymbolDataset(json_path, image_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    return loader




class PatchEmbedding(nn.Module):
    def __init__(self, image_size=800, patch_size=32, in_channels=3, embed_dim=512):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                  kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.projection(x)  # (B, E, H', W')
        x = rearrange(x, 'b e h w -> b (h w) e')  # (B, N, E)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, E = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(dots.softmax(dim=-1))
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SymbolEncoder(nn.Module):
    def __init__(self, image_size=800, patch_size=32, in_channels=3, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, 
                 dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Add classification token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x[:, 0]  # Return only the CLS token embedding

class SymbolMatcher(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        # self.temperature = 0.07  # Fixed temperature
    
    def forward(self, reference, candidates):
        ref_embedding = self.encoder(reference)  # (B, E)
        cand_embeddings = self.encoder(candidates)  # (B*4, E) or (N, E) for inference
        
        # Handle different batch shapes for training vs inference
        if reference.size(0) == 1:  # Inference mode
            similarity = F.cosine_similarity(
                ref_embedding.unsqueeze(0),  # (1, 1, E)
                cand_embeddings.unsqueeze(0),  # (1, N, E)
                dim=2
            ) / self.temperature
        else:  # Training mode
            B = candidates.shape[0] // 4
            cand_embeddings = cand_embeddings.view(B, 4, -1)  # Reshape to (B, 4, E)
            similarity = F.cosine_similarity(
                ref_embedding.unsqueeze(1),  # (B, 1, E)
                cand_embeddings,  # (B, 4, E)
                dim=2
            ) / self.temperature
        
        return similarity
    # def forward(self, reference, candidates):
    #     B = candidates.shape[0] // 4  # Divide by number of candidates per reference
    #     ref_embedding = self.encoder(reference)  # (B, E)
    #     cand_embeddings = self.encoder(candidates)  # (B*4, E)
    #     cand_embeddings = cand_embeddings.view(B, 4, -1)  # Reshape to (B, 4, E)
        
    #     similarity = F.cosine_similarity(
    #         ref_embedding.unsqueeze(1),  # (B, 1, E)
    #         cand_embeddings,  # (B, 4, E)
    #         dim=2
    #     ) / self.temperature
        
    #     return similarity

def create_symbol_matcher(image_size=800, patch_size=32, in_channels=3,
                         embed_dim=768, depth=12, num_heads=12):
    encoder = SymbolEncoder(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads
    )
    return SymbolMatcher(encoder)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        

# Training utilities
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, similarity, positive_idx):
        positive_idx = positive_idx.squeeze(-1)
        loss = F.cross_entropy(similarity, positive_idx)
        
        # Add margin penalty
        max_negative = torch.max(
            similarity - torch.eye(similarity.size(0)).to(similarity.device) * 1e9,
            dim=1
        )[0]
        margin_loss = F.relu(max_negative - similarity.diagonal() + self.margin)
        
        return loss + margin_loss.mean()

def train_step(model, optimizer, reference, candidates, positive_idx):
    optimizer.zero_grad()
    
    # Forward pass
    similarity = model(reference, candidates)
    
    # Compute loss
    loss = ContrastiveLoss()(similarity, positive_idx)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def inference(model, reference, candidates, threshold=0.5):
    with torch.no_grad():
        similarity = model(reference, candidates)
        predictions = torch.sigmoid(similarity) > threshold
    return predictions



def train_model(json_path=r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco\train\_annotations.coco.json",
                image_dir=r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco\train",
                output_dir=r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco",
                num_epochs=10, batch_size=28, learning_rate=1e-4, device='cuda', accumulation_steps=4):
    
    # model = create_symbol_matcher().to(device)
    # 1. Create model with reduced size
    model = create_symbol_matcher(
        image_size=800,  # Reduced from 800
        patch_size=32,
        embed_dim=512,   # Reduced from 768
        depth=6,         # Reduced from 12
        num_heads=8      # Reduced from 12
    ).to(device)
    
    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # train_loader = get_dataloaders(json_path, image_dir, batch_size)
    train_loader = get_dataloaders(json_path, image_dir, batch_size)

    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        optimizer.zero_grad()
        scaler = GradScaler()
        for batch in pbar:
            reference = batch['reference'].to(device, dtype=torch.bfloat16)
            candidates = batch['candidates'].reshape(-1, 3, 800, 800).to(device, dtype=torch.bfloat16)
            positive_idx = batch['positive_idx'].to(device)
            optimizer.zero_grad()

            with autocast():
                similarity = model(reference, candidates)
                loss = ContrastiveLoss()(similarity, positive_idx)
                loss = loss / accumulation_steps  # Scale loss
            
            scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            if (batch + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            epoch_loss += loss
            pbar.set_postfix({'loss': f'{loss:.4f}'})
            if batch % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}')
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(output_dir, 'best_model.pth'))
            
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    
    return model

def run_inference(model_path, reference_image, candidate_images, device='cuda'):
    # Load model
    model = create_symbol_matcher().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # Prepare images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    reference = transform(reference_image).unsqueeze(0).to(device)
    candidates = torch.stack([transform(img) for img in candidate_images]).to(device)
    
    # Run inference
    with torch.no_grad():
        similarity = model(reference, candidates)
        predictions = torch.sigmoid(similarity) > 0.5
        confidence = torch.sigmoid(similarity)
    
    return predictions.cpu().numpy(), confidence.cpu().numpy()

if __name__ == "__main__":
    # Train
    model = train_model()
    reference_image = 'icon_switch.png'
    candidate_images = 'FlrPln.png'
    # Example inference
    model_path = r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco\best_model.pth"
    from PIL import Image

    # Load your images
    reference_image = Image.open('icon_switch.png').convert('RGB')
    candidate_images = [
        Image.open(r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco\train\PC-set-E-shts-7-20-22-21-11-2-1_png.rf.45da7a961b4480a5e9259aac071a952b.jpg").convert('RGB'),
        # Image.open("path/to/candidate2.jpg"),
        # Image.open("path/to/candidate3.jpg"),
        # Image.open("path/to/candidate4.jpg")
    ]
    predictions, confidence = run_inference(model_path, reference_image, candidate_images)
    img:Image = Image.open(r"M:\new downloads\elec-stuff.v22-2025-01-25-isolated-objects.coco\train\PC-set-E-shts-7-20-22-21-11-2-1_png.rf.45da7a961b4480a5e9259aac071a952b.jpg").convert('RGB'),
    
    # Interpret results
    for i, (pred, conf) in enumerate(zip(predictions.flatten(), confidence.flatten())):
        print(f"Candidate {i}: Match={bool(pred)}, Confidence={float(conf):.2f}")