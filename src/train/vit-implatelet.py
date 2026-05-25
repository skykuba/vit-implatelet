# %% [markdown]
# # ViT-B/16 Cancer Classification - Kaggle Notebook
# 
# **Progressive Fine-tuning Strategy for small medical datasets**
# 
# Training strategy:
# - **Phase 1**: Training only the classifier head (frozen encoder)
# - **Phase 2**: Unfreezing last N transformer blocks + continuing training with lower LR
# 
# ---
# 
# ## 1. Installation and Library Imports
# %%
import os
import sys
import wandb

def detect_env():
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "Kaggle"
    if "google.colab" in sys.modules:
        return "Google Colab"
    return "Local or other environment"

ENV = detect_env()
print(f"Detected environment: {ENV}")

if ENV == "Kaggle":
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
    os.environ["KAGGLE_USERNAME"] = user_secrets.get_secret("KAGGLE_USERNAME")
    print("Loaded secrets from Kaggle")

elif ENV == "Google Colab":
    try:
        from google.colab import userdata
        os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        print("Loaded WANDB_API_KEY from Colab secrets")
    except Exception as e:
        print("Google Colab module not available or WANDB_API_KEY not found in secrets", e)

else:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded .env file")
    except Exception as e:
        print("python-dotenv not installed or .env not found", e)

wandb.login()
# %%
import subprocess
import sys
from datetime import datetime

run_start = datetime.now().strftime("%Y-%m-%d_%H-%M")

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

wandb.login(key=os.getenv("WANDB_API_KEY"))
# %% [markdown]
# ## 2. Kaggle Configuration
# 
# Configuration adapted for Kaggle environment:
# - Paths: `/kaggle/input/` for input data, `/kaggle/working/` for results
# - GPU: CUDA (P100/T4/V100 depending on accelerator)
# - Optimal parameters for small medical datasets
# %%
class Config:
    if ENV == "Kaggle":
        DATA_DIR = "/kaggle/input/datasets/skykuba/implatelet/KEGG_Pathway_Image/Images"  
        SAVE_DIR = "/kaggle/working/weights"
    elif ENV == "Google Colab":
        import kagglehub
        path = kagglehub.dataset_download("skykuba/implatelet")
        DATA_DIR = f"{path}/KEGG_Pathway_Image/Images"
        SAVE_DIR = "/content/weights"
    else:
        DATA_DIR = "KEGG_Pathway_Image/Images"
        SAVE_DIR = "weights_local"
    
    ENV = ENV
    NUM_CLASSES = 2
    PRETRAINED = True
    
    IMG_SIZE = 224
    ORIGINAL_SIZE = (373, 259) 
    
    BATCH_SIZE = 32  
    NUM_WORKERS = 0 
    
    # Phase 1: Training only the classifier head (frozen encoder)
    PHASE1_EPOCHS = 100
    PHASE1_LR = 1e-3
    
    # Phase 2: Fine-tuning last blocks + head
    PHASE2_EPOCHS = 100
    PHASE2_LR = 1e-5
    UNFREEZE_BLOCKS = 4  # Unfreeze last N transformer blocks
    
    # ========================
    # REGULARIZATION
    # ========================
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.1
    LABEL_SMOOTHING = 0.1
    
    # ========================
    # NORMALIZATION
    # ========================
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # ========================
    # AUGMENTATIONS
    # ========================
    AUG_HFLIP_ENABLED = True
    AUG_HFLIP_PROB = 0.5
    
    AUG_VFLIP_ENABLED = True
    AUG_VFLIP_PROB = 0.5
    
    AUG_ROTATION_ENABLED = True
    AUG_ROTATION_DEGREES = [0, 90, 180, 270] # Discrete rotations
    
    AUG_RESIZED_CROP_ENABLED = True
    AUG_RESIZED_CROP_SCALE = (0.8, 1.0)
    AUG_RESIZED_CROP_RATIO = (0.9, 1.1)
    
    AUG_COLOR_JITTER_ENABLED = True
    AUG_COLOR_JITTER_BRIGHTNESS = 0.2
    AUG_COLOR_JITTER_CONTRAST = 0.2
    
    AUG_RANDOM_ERASING_ENABLED = True
    AUG_RANDOM_ERASING_PROB = 0.1
    AUG_RANDOM_ERASING_SCALE = (0.02, 0.1)
    
    AUG_GAUSSIAN_NOISE_ENABLED = True
    AUG_GAUSSIAN_NOISE_MEAN = 0.0
    AUG_GAUSSIAN_NOISE_STD = 0.05
    
    AUG_POINT_DROPOUT_ENABLED = True
    AUG_POINT_DROPOUT_PROB = 0.05 # Probability of dropping a pixel
    
    # ========================
    # DATA SPLIT
    # ========================
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    SEED = 42
    
    # ========================
    # DEVICE
    # ========================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Display configuration
print("TRAINING CONFIGURATION")
print("=" * 50)
print(f"Environment: {Config.ENV}")
print(f"Data directory: {Config.DATA_DIR}")
print(f"Save directory: {Config.SAVE_DIR}")
print(f"Device: {Config.DEVICE}")
print(f"Batch size: {Config.BATCH_SIZE}")
print(f"Phase 1: {Config.PHASE1_EPOCHS} epochs, LR={Config.PHASE1_LR}")
print(f"Phase 2: {Config.PHASE2_EPOCHS} epochs, LR={Config.PHASE2_LR}")
print("=" * 50)
# %%
def get_kaggle_username():
    return (os.environ.get("KAGGLE_USERNAME")
    or os.environ.get("KAGGLE_USER")
    or "unknown_user")

run = wandb.init(
    entity="pg-ai-team",
    project="vit-implatelet",
    name=get_kaggle_username()+"_"+run_start,
    config={
        "architecture": "ViT-B/16",
        "dataset": "skykuba/implatelet",
        "dataset_version": "v1",
        "num_classes": Config.NUM_CLASSES,
        "img_size": Config.IMG_SIZE,
        "original_size": Config.ORIGINAL_SIZE,
        "batch_size": Config.BATCH_SIZE,
        "num_workers": Config.NUM_WORKERS,
        "phase1_epochs": Config.PHASE1_EPOCHS,
        "phase1_lr": Config.PHASE1_LR,
        "phase2_epochs": Config.PHASE2_EPOCHS,
        "phase2_lr": Config.PHASE2_LR,
        "unfreeze_blocks": Config.UNFREEZE_BLOCKS,
        "weight_decay": Config.WEIGHT_DECAY,
        "dropout": Config.DROPOUT,
        "label_smoothing": Config.LABEL_SMOOTHING,
        
        "normalize_mean": Config.NORMALIZE_MEAN,
        "normalize_std": Config.NORMALIZE_STD,
        
        # Augmentations
        "aug_hflip_enabled": Config.AUG_HFLIP_ENABLED,
        "aug_hflip_prob": Config.AUG_HFLIP_PROB,
        "aug_vflip_enabled": Config.AUG_VFLIP_ENABLED,
        "aug_vflip_prob": Config.AUG_VFLIP_PROB,
        "aug_rotation_enabled": Config.AUG_ROTATION_ENABLED,
        "aug_rotation_degrees": Config.AUG_ROTATION_DEGREES,
        "aug_resized_crop_enabled": Config.AUG_RESIZED_CROP_ENABLED,
        "aug_resized_crop_scale": Config.AUG_RESIZED_CROP_SCALE,
        "aug_resized_crop_ratio": Config.AUG_RESIZED_CROP_RATIO,
        "aug_color_jitter_enabled": Config.AUG_COLOR_JITTER_ENABLED,
        "aug_color_jitter_brightness": Config.AUG_COLOR_JITTER_BRIGHTNESS,
        "aug_color_jitter_contrast": Config.AUG_COLOR_JITTER_CONTRAST,
        "aug_random_erasing_enabled": Config.AUG_RANDOM_ERASING_ENABLED,
        "aug_random_erasing_prob": Config.AUG_RANDOM_ERASING_PROB,
        "aug_random_erasing_scale": Config.AUG_RANDOM_ERASING_SCALE,
        "aug_gaussian_noise_enabled": Config.AUG_GAUSSIAN_NOISE_ENABLED,
        "aug_gaussian_noise_mean": Config.AUG_GAUSSIAN_NOISE_MEAN,
        "aug_gaussian_noise_std": Config.AUG_GAUSSIAN_NOISE_STD,
        "aug_point_dropout_enabled": Config.AUG_POINT_DROPOUT_ENABLED,
        "aug_point_dropout_prob": Config.AUG_POINT_DROPOUT_PROB,
        
        "val_split": Config.VAL_SPLIT,
        "test_split": Config.TEST_SPLIT,
        "seed": Config.SEED,
        "device": str(Config.DEVICE),
        "data_dir": Config.DATA_DIR,
        "save_dir": Config.SAVE_DIR,
        "pretrained": Config.PRETRAINED,
    },
)
# %% [markdown]
# ## 3. Setting Random Seed
# 
# Ensuring result reproducibility by setting seeds for all random number generators.
# %%
def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For full reproducibility (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set seed
set_seed(Config.SEED)
print(f"Seed set to: {Config.SEED}")
# %% [markdown]
# ## 4. Dataset Class Definition
# 
# `KEGGPathwayDataset` class loads KEGG Pathway images and their labels (Malignant/nonMalignant).
# %%
class KEGGPathwayDataset(Dataset):
    """Dataset for KEGG Pathway images for cancer classification."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

print("KEGGPathwayDataset class defined")
# %% [markdown]
# ## 5. Data Transformations
# 
# - **Training**: Augmentations (flip, rotation, color jitter, random erasing) + ImageNet normalization
# - **Validation/Test**: Only resize + ImageNet normalization
# %%
import torchvision.transforms.functional as F

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class PointDropout(object):
    def __init__(self, p=0.01):
        self.p = p
        
    def __call__(self, tensor):
        mask = torch.rand(tensor.size()) > self.p
        return tensor * mask
    
    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'

class RandomDiscreteRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)

def get_data_transforms():
    train_transform_list = []
    
    # Geometric Augmentations
    if Config.AUG_RESIZED_CROP_ENABLED:
        train_transform_list.append(transforms.RandomResizedCrop(
            Config.IMG_SIZE, 
            scale=Config.AUG_RESIZED_CROP_SCALE, 
            ratio=Config.AUG_RESIZED_CROP_RATIO
        ))
    else:
        train_transform_list.append(transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)))

    if Config.AUG_HFLIP_ENABLED:
        train_transform_list.append(transforms.RandomHorizontalFlip(p=Config.AUG_HFLIP_PROB))
        
    if Config.AUG_VFLIP_ENABLED:
        train_transform_list.append(transforms.RandomVerticalFlip(p=Config.AUG_VFLIP_PROB))
        
    if Config.AUG_ROTATION_ENABLED:
        train_transform_list.append(RandomDiscreteRotation(Config.AUG_ROTATION_DEGREES))

    # Color/Signal Augmentations
    if Config.AUG_COLOR_JITTER_ENABLED:
        train_transform_list.append(transforms.ColorJitter(
            brightness=Config.AUG_COLOR_JITTER_BRIGHTNESS,
            contrast=Config.AUG_COLOR_JITTER_CONTRAST
        ))
        
    # ToTensor must be before Noise/Erasing
    train_transform_list.append(transforms.ToTensor())
    
    # Noise and Dropout (applied to Tensors)
    if Config.AUG_GAUSSIAN_NOISE_ENABLED:
        train_transform_list.append(AddGaussianNoise(
            mean=Config.AUG_GAUSSIAN_NOISE_MEAN,
            std=Config.AUG_GAUSSIAN_NOISE_STD
        ))
        
    if Config.AUG_POINT_DROPOUT_ENABLED:
        train_transform_list.append(PointDropout(p=Config.AUG_POINT_DROPOUT_PROB))

    if Config.AUG_RANDOM_ERASING_ENABLED:
        train_transform_list.append(transforms.RandomErasing(
            p=Config.AUG_RANDOM_ERASING_PROB,
            scale=Config.AUG_RANDOM_ERASING_SCALE
        ))

    # Normalization
    train_transform_list.append(transforms.Normalize(
        mean=Config.NORMALIZE_MEAN,
        std=Config.NORMALIZE_STD
    ))

    train_transform = transforms.Compose(train_transform_list)
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=Config.NORMALIZE_MEAN,
            std=Config.NORMALIZE_STD
        ),
    ])
    
    return train_transform, val_transform

print("Data transforms defined with configurable augmentations")
# %% [markdown]
# ## 6. Data Loading and Splitting
# 
# Loading PNG images from directory, parsing labels from filenames, and stratified split into train/val/test.
# %%
def load_dataset():
    """Load and split the dataset."""
    image_paths = []
    labels = []
    
    data_dir = Config.DATA_DIR
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory {data_dir} does not exist!")

        
        # List available datasets in Kaggle
        if os.path.exists('/kaggle/input'):
            print("\nAvailable datasets in /kaggle/input:")
            for item in os.listdir('/kaggle/input'):
                print(f"   - {item}")
        return None, None, None
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            filepath = os.path.join(data_dir, filename)
            
            # Parse label from filename
            if filename.startswith('Malignant_'):
                label = 1  # Cancer
            elif filename.startswith('nonMalignant_'):
                label = 0  # No cancer
            else:
                continue
                
            image_paths.append(filepath)
            labels.append(label)
    
    print(f"Loaded {len(image_paths)} images")
    print(f"Class distribution: {Counter(labels)}")
    
    # Split: train / val / test
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, 
        test_size=(Config.VAL_SPLIT + Config.TEST_SPLIT),
        stratify=labels,
        random_state=Config.SEED
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=Config.TEST_SPLIT / (Config.VAL_SPLIT + Config.TEST_SPLIT),
        stratify=y_temp,
        random_state=Config.SEED
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Load data
train_data, val_data, test_data = load_dataset()
# %% [markdown]
# ## 7. Weighted Sampler for Unbalanced Classes
# 
# Creating a sampler with weights inversely proportional to class frequencies to balance training.
# %%
def get_weighted_sampler(labels):
    """Create weighted sampler for unbalanced classes."""
    class_counts = Counter(labels)
    total = len(labels)
    
    # Weights inversely proportional to frequency
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

print("get_weighted_sampler function defined")
# %% [markdown]
# ## 8. ViT Classifier Model Definition
# 
# ViT-B/16 model with pretrained ImageNet weights and custom classification head.
# 
# Key methods:
# - `freeze_encoder()` - freezes encoder for Phase 1
# - `unfreeze_last_blocks(n)` - unfreezes last n transformer blocks for Phase 2
# %%
class ViTClassifier(nn.Module):
    """ViT-B/16 with custom classification head."""
    
    def __init__(self, num_classes=2, dropout=0.1, pretrained=True):
        super().__init__()
        
        # Load pretrained ViT-B/16
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.vit = vit_b_16(weights=weights)
            print("Loaded pretrained ImageNet weights")
        else:
            self.vit = vit_b_16(weights=None)
        
        # Get hidden dimension
        hidden_dim = self.vit.heads.head.in_features  
        
        # Replace classification head with custom one
        self.vit.heads = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.vit(x)
    
    def freeze_encoder(self):
        """Freeze all encoder layers (for Phase 1)."""
        for param in self.vit.parameters():
            param.requires_grad = False
        # Unfreeze head
        for param in self.vit.heads.parameters():
            param.requires_grad = True
            
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Encoder frozen. Trainable parameters: {trainable:,}")
    
    def unfreeze_last_blocks(self, n_blocks=4):
        """Unfreeze last N transformer blocks (for Phase 2)."""
        # First freeze everything
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Unfreeze head
        for param in self.vit.heads.parameters():
            param.requires_grad = True
        
        # Unfreeze last N encoder blocks
        total_blocks = len(self.vit.encoder.layers)
        for i in range(total_blocks - n_blocks, total_blocks):
            for param in self.vit.encoder.layers[i].parameters():
                param.requires_grad = True
        
        # Unfreeze layer norm
        for param in self.vit.encoder.ln.parameters():
            param.requires_grad = True
            
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Last {n_blocks} blocks unfrozen. Trainable parameters: {trainable:,}")
    
    def unfreeze_all(self):
        """Unfreeze entire model."""
        for param in self.vit.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"All layers unfrozen. Trainable parameters: {trainable:,}")

print("ViTClassifier class defined")
# %% [markdown]
# ## 9. Training and Validation Functions
# 
# - `train_epoch()` - training one epoch with gradient clipping
# - `validate()` - validation with loss, accuracy, and AUC-ROC calculation
# %%
def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    # Calculate AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return epoch_loss, epoch_acc, auc, all_preds, all_labels

print("train_epoch and validate functions defined")
# %% [markdown]
# ## 10. Early Stopping
# 
# Early Stopping class to prevent overfitting by monitoring validation loss.
# %%
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=40, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, phase_name, save_path):
    """Train one phase."""
    print(f"\n{'='*60}")
    print(f"Starting {phase_name}")
    print(f"{'='*60}")
    
    best_val_acc = 0.0
    best_val_auc = 0.0
    early_stopping = EarlyStopping()
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation
        val_loss, val_acc, val_auc, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Logging
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")
        
        # Log to wandb
        run.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc
        })
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_auc': val_auc,
            }, save_path)
            print(f"Best model saved (AUC: {val_auc:.4f})")
            
            # Log best model to wandb
            run.log({"best_val_auc": best_val_auc, "best_val_acc": best_val_acc})
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n{phase_name} completed!")
    print(f"Best Val Acc: {best_val_acc:.2f}%, Best Val AUC: {best_val_auc:.4f}")
    
    return history

print("EarlyStopping and train_phase defined")
# %% [markdown]
# ## 11. Phase 1: Training Classifier Head
# 
# **Phase 1 Strategy:**
# - Frozen ViT encoder (all transformer blocks)
# - Training only custom classification head
# - Higher learning rate (1e-3)
# - ReduceLROnPlateau scheduler
# %%
# Check if data was loaded correctly
if train_data is None:
    raise ValueError("Data not loaded! Check path in Config.DATA_DIR")

X_train, y_train = train_data
X_val, y_val = val_data
X_test, y_test = test_data

# Create save directory
os.makedirs(Config.SAVE_DIR, exist_ok=True)

# Get transforms
train_transform, val_transform = get_data_transforms()

# Create datasets
train_dataset = KEGGPathwayDataset(X_train, y_train, train_transform)
val_dataset = KEGGPathwayDataset(X_val, y_val, val_transform)
test_dataset = KEGGPathwayDataset(X_test, y_test, val_transform)

# Create weighted sampler
train_sampler = get_weighted_sampler(y_train)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    sampler=train_sampler,
    num_workers=Config.NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
    pin_memory=True
)

print(f"DataLoaders created")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")
# %%
# Create model
print("Creating ViT-B/16 model...")
model = ViTClassifier(
    num_classes=Config.NUM_CLASSES,
    dropout=Config.DROPOUT,
    pretrained=Config.PRETRAINED
)
model = model.to(Config.DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Class weights for loss function
class_counts = Counter(y_train)
total = len(y_train)
class_weights = torch.tensor([
    total / class_counts[0],
    total / class_counts[1]
], dtype=torch.float32).to(Config.DEVICE)
class_weights = class_weights / class_weights.sum()  # Normalization

print(f"Class weights: {class_weights.cpu().numpy()}")

criterion = nn.CrossEntropyLoss(
    label_smoothing=Config.LABEL_SMOOTHING
)
# %%
# ============================================================================
# PHASE 1: Training only the classifier head
# ============================================================================
model.freeze_encoder()

optimizer_phase1 = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=Config.PHASE1_LR,
    weight_decay=Config.WEIGHT_DECAY
)

scheduler_phase1 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_phase1, mode='min', factor=0.5, patience=3
)

history_phase1 = train_phase(
    model, train_loader, val_loader, criterion,
    optimizer_phase1, scheduler_phase1, Config.DEVICE,
    Config.PHASE1_EPOCHS, "Phase 1: Training Classifier Head",
    os.path.join(Config.SAVE_DIR, "vit_phase1_best.pth")
)
# %% [markdown]
# ## 12. Phase 2: Fine-tuning Last Blocks
# 
# **Phase 2 Strategy:**
# - Unfreezing last 4 transformer blocks + head
# - Lower learning rate (1e-5) for stable fine-tuning
# - CosineAnnealingWarmRestarts scheduler
# - Continuing from best model of Phase 1
# %%
# ============================================================================
# PHASE 2: Fine-tuning last blocks + head
# ============================================================================

# Load best model from Phase 1
checkpoint = torch.load(os.path.join(Config.SAVE_DIR, "vit_phase1_best.pth"), weights_only=False)

# If we used DataParallel, model has different keys in state_dict
if isinstance(model, nn.DataParallel):
    # Add 'module.' prefix if missing
    state_dict = checkpoint['model_state_dict']
    if not any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
else:
    model.load_state_dict(checkpoint['model_state_dict'])

print(f"Loaded best model from Phase 1 (AUC: {checkpoint['val_auc']:.4f})")

# Unfreeze last blocks
base_model = model.module if isinstance(model, nn.DataParallel) else model
base_model.unfreeze_last_blocks(Config.UNFREEZE_BLOCKS)

optimizer_phase2 = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=Config.PHASE2_LR,
    weight_decay=Config.WEIGHT_DECAY
)

scheduler_phase2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_phase2, T_0=5, T_mult=2
)

history_phase2 = train_phase(
    model, train_loader, val_loader, criterion,
    optimizer_phase2, scheduler_phase2, Config.DEVICE,
    Config.PHASE2_EPOCHS, f"Phase 2: Fine-tuning Last {Config.UNFREEZE_BLOCKS} Blocks",
    os.path.join(Config.SAVE_DIR, "vit_phase2_best.pth")
)
# %% [markdown]
# ## 13. Evaluation on Test Set
# 
# Final evaluation of best model on test set with calculation of:
# - Accuracy, AUC-ROC
# - Sensitivity (Recall for Malignant class)
# - Specificity (Recall for nonMalignant class)
# - Confusion Matrix and Classification Report
# %%
def evaluate_model(model, test_loader, device):
    """Final evaluation on test set."""
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_auc, preds, labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, preds, 
                               target_names=['nonMalignant', 'Malignant']))
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Sensitivity and Specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    
    print(f"\nSensitivity (Recall for Malignant): {sensitivity:.2f}%")
    print(f"Specificity (Recall for nonMalignant): {specificity:.2f}%")
    
    # Log test results to wandb
    run.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=labels, 
            preds=preds,
            class_names=['nonMalignant', 'Malignant']
        )
    })
    
    return test_acc, test_auc, cm

# Load best model from Phase 2
best_checkpoint = torch.load(os.path.join(Config.SAVE_DIR, "vit_phase2_best.pth"), weights_only=False)

# Handle DataParallel
if isinstance(model, nn.DataParallel):
    state_dict = best_checkpoint['model_state_dict']
    if not any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
else:
    model.load_state_dict(best_checkpoint['model_state_dict'])

print(f"Loaded best model from Phase 2 (AUC: {best_checkpoint['val_auc']:.4f})")

# Evaluation
test_acc, test_auc, cm = evaluate_model(model, test_loader, Config.DEVICE)
# %%
# Save final model
final_path = os.path.join(Config.SAVE_DIR, "vit_final.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'test_acc': test_acc,
    'test_auc': test_auc,
    'config': {
        'num_classes': Config.NUM_CLASSES,
        'img_size': Config.IMG_SIZE,
        'unfreeze_blocks': Config.UNFREEZE_BLOCKS,
    }
}, final_path)
print(f"\nFinal model saved to: {final_path}")

# List saved files
print("\nSaved files:")
for f in os.listdir(Config.SAVE_DIR):
    filepath = os.path.join(Config.SAVE_DIR, f)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"   - {f} ({size_mb:.1f} MB)")
# %% [markdown]
# ## 14. Training Results Visualization
# 
# Plots showing training progress for both phases:
# - Loss (train vs validation)
# - Accuracy (train vs validation)  
# - AUC-ROC (validation)
# %%
def plot_training_history(history_phase1, history_phase2, save_path=None):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Combine histories
    epochs_p1 = len(history_phase1['train_loss'])
    
    all_train_loss = history_phase1['train_loss'] + history_phase2['train_loss']
    all_val_loss = history_phase1['val_loss'] + history_phase2['val_loss']
    all_train_acc = history_phase1['train_acc'] + history_phase2['train_acc']
    all_val_acc = history_phase1['val_acc'] + history_phase2['val_acc']
    all_val_auc = history_phase1['val_auc'] + history_phase2['val_auc']
    
    epochs = range(1, len(all_train_loss) + 1)
    
    # Loss plot
    axes[0].plot(epochs, all_train_loss, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, all_val_loss, 'r-', label='Val', linewidth=2)
    axes[0].axvline(x=epochs_p1, color='g', linestyle='--', label='Phase 2', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, all_train_acc, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, all_val_acc, 'r-', label='Val', linewidth=2)
    axes[1].axvline(x=epochs_p1, color='g', linestyle='--', label='Phase 2', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # AUC plot
    axes[2].plot(epochs, all_val_auc, 'r-', label='Val AUC', linewidth=2)
    axes[2].axvline(x=epochs_p1, color='g', linestyle='--', label='Phase 2', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('AUC', fontsize=12)
    axes[2].set_title('Validation AUC-ROC', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to: {save_path}")
    
    plt.show()

# Plot results
plot_training_history(
    history_phase1, 
    history_phase2,
    os.path.join(Config.SAVE_DIR, "training_curves.png")
)
# %%
# Confusion Matrix visualization
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['nonMalignant', 'Malignant'],
            yticklabels=['nonMalignant', 'Malignant'],
            annot_kws={'size': 16})
ax.set_xlabel('Predicted', fontsize=14)
ax.set_ylabel('Actual', fontsize=14)
ax.set_title('Confusion Matrix', fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(Config.SAVE_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nConfusion matrix saved to: {os.path.join(Config.SAVE_DIR, 'confusion_matrix.png')}")
# %% [markdown]
# ## Summary
# 
# Training completed! Model trained using progressive fine-tuning:
# 
# 1. **Phase 1** - Training classifier head with frozen encoder
# 2. **Phase 2** - Fine-tuning last transformer blocks
# 
# Result files located in `/kaggle/working/weights/`:
# - `vit_phase1_best.pth` - best model from Phase 1
# - `vit_phase2_best.pth` - best model from Phase 2  
# - `vit_final.pth` - final model
# - `training_curves.png` - training plots
# - `confusion_matrix.png` - confusion matrix
# %%
# Results summary
print("=" * 60)
print("TRAINING COMPLETED!")
print("=" * 60)
print(f"\nTEST SET RESULTS:")
print(f"   • Test Accuracy: {test_acc:.2f}%")
print(f"   • Test AUC-ROC:  {test_auc:.4f}")
print(f"\nSAVED MODELS:")
print(f"   • {os.path.join(Config.SAVE_DIR, 'vit_final.pth')}")
print("\n" + "=" * 60)

# For Kaggle - to download the model, use:
# from kaggle_secrets import UserSecretsClient
# or just download from /kaggle/working/weights/

# Finish the run and upload any remaining data
run.finish()