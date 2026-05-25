# %% [markdown]
# 1. IMPORTS
# %%
import os
os.environ["WANDB_START_METHOD"] = "thread"
import wandb
# import kaggle_secrets import UserSecretsClient
import os
import numpy as np
import torch
from timm import create_model
import timm
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import random
import torchvision.transforms.functional as F
from PIL import Image
from datetime import datetime

# user_secrets = UserSecretsClient()
# secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")
# os.environ["WANDB_API_KEY"] = secret_value_0
# secret_value_1 = user_secrets.get_secret("KAGGLE_USERNAME")
# os.environ["KAGGLE_USERNAME"] = secret_value_1


os.environ["WANDB_API_KEY"] = " "
os.environ["KAGGLE_USERNAME"] = " "

run_start = datetime.now().strftime("%Y-%m-%d_%H-%M")
wandb.login(key=os.getenv("WANDB_API_KEY"))
# %% [markdown]
# 2. CONFIG
# %%
class Config:
    if os.path.exists('/kaggle'):
        DATA_DIR = "/kaggle/input/datasets/skykuba/implatelet/KEGG_Pathway_Image/Images"
        SAVE_DIR = "/kaggle/working/weights"
    else:
        DATA_DIR = "KEGG_Pathway_Image/Images"
        SAVE_DIR = "weights_kaggle"


    NUM_CLASSES = 1
    PRETRAINED = False

    IMG_SIZE = 224
    ORIGINAL_SIZE = (373, 259)

    BATCH_SIZE = 32
    NUM_WORKERS = 0

    # Phase 1: Training only the classifier head (frozen encoder)
    PHASE1_EPOCHS = 20
    PHASE1_LR = 0.0003

    # Phase 2: Fine-tuning last blocks + head
    PHASE2_EPOCHS = 300
    PHASE2_LR = 0.000003
    UNFREEZE_BLOCKS = 6  # Unfreeze last N transformer blocks

    # ========================
    # REGULARIZATION
    # ========================
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.5
    LABEL_SMOOTHING = 0.0

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

# run = wandb.init(
#     entity="pg-ai-team",
#     project="vit-implatelet",
#     name=get_kaggle_username()+"_"+run_start,
#     config={
#         "architecture": "ViT-B/16",
#         "dataset": "skykuba/implatelet",
#         "dataset_version": "v1",
#         "num_classes": Config.NUM_CLASSES,
#         "img_size": Config.IMG_SIZE,
#         "original_size": Config.ORIGINAL_SIZE,
#         "batch_size": Config.BATCH_SIZE,
#         "num_workers": Config.NUM_WORKERS,
#         "phase1_epochs": Config.PHASE1_EPOCHS,
#         "phase1_lr": Config.PHASE1_LR,
#         "phase2_epochs": Config.PHASE2_EPOCHS,
#         "phase2_lr": Config.PHASE2_LR,
#         "unfreeze_blocks": Config.UNFREEZE_BLOCKS,
#         "weight_decay": Config.WEIGHT_DECAY,
#         "dropout": Config.DROPOUT,
#         "label_smoothing": Config.LABEL_SMOOTHING,
#         "val_split": Config.VAL_SPLIT,
#         "test_split": Config.TEST_SPLIT,
#         "seed": Config.SEED,
#         "device": str(Config.DEVICE),
#         "data_dir": Config.DATA_DIR,
#         "save_dir": Config.SAVE_DIR,
#         "pretrained": Config.PRETRAINED,
#     },
# )
# %% [markdown]
# 3. SEED
# %%
def set_seed(SEED):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed
set_seed(Config.SEED)
print(f"Seed set to: {Config.SEED}")
# %% [markdown]
# 4. MODEL
# %%
base_model = create_model('vit_tiny_patch16_224', pretrained=False, in_chans=1, patch_size=(1,224))

# for param in base_model.parameters():
#     param.requires_grad = False

# for name, param in base_model.named_parameters():
#     if "norm." in name or\
#     "blocks.11" in name:
#         print(name)
#         param.requires_grad = True

# Modify the classifier head
num_classes = Config.NUM_CLASSES
in_features = base_model.head.in_features
base_model.head = nn.Sequential(
    nn.Linear(in_features, 16),
    nn.BatchNorm1d(16),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(16, num_classes)
)

# Move model to device
model = base_model.to(Config.DEVICE)
model
# %% [markdown]
# 5. DATASET
# %%
class MalignantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Pobieramy listę wszystkich plików
        self.file_names = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB') # Ładujemy jako RGB dla pewności

        # Przypisanie etykiety na podstawie nazwy pliku
        # 1 dla malignant, 0 dla nonmalignant
        if 'nonmalignant' in img_name.lower():
            label = 0
        elif 'malignant' in img_name.lower():
            label = 1
        else:
            label = 0 # Wartość domyślna, jeśli nazwa nie pasuje

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda img: F.crop(img, top=0, left=0, height=224, width=224)),
    transforms.ToTensor()
])

# Tworzymy pełny obiekt datasetu
full_dataset = MalignantDataset(root_dir=Config.DATA_DIR, transform=transform)

# Podział (80% train, 10% val, 10% test)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(Config.SEED)
)

# Loadery
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

weight = torch.tensor([1.1]).to(Config.DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.5)
# %% [markdown]
# 6. TRAINING
# %%
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_probs = []
    train_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images) # To są logity
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Obliczanie celności dla 1 neuronu
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        train_probs.append(probs.cpu())
        train_labels.append(labels.cpu())

    train_loss = running_loss / len(train_loader)
    train_probs = torch.cat(train_probs).detach().numpy()
    train_labels = torch.cat(train_labels).detach().numpy()

    if len(np.unique(train_labels)) > 1:
        train_auc = roc_auc_score(train_labels, train_probs)
    else:
        train_auc = 0


    return train_loss, 100. * correct / total, train_auc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # Przygotowanie etykiet pod BCEWithLogitsLoss
            labels_float = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels_float)

            running_loss += loss.item()

            # Prawdopodobieństwo i przewidywania
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            total += labels.size(0)
            correct += preds.eq(labels_float).sum().item()

            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    # OBLICZENIE BRAKUJĄCYCH ZMIENNYCH:
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    all_preds = [1 if p > 0.5 else 0 for p in all_probs]

    # Obliczanie AUC
    try:
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.5
    except:
        auc = 0.0

    return epoch_loss, epoch_acc, auc

def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, phase_name, save_path):
    print(f"\n{'='*60}")
    print(f"Starting {phase_name}")
    print(f"{'='*60}")

    best_val_acc = 0.0
    best_val_auc = 0.0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training
        train_loss, train_acc, train_auc= train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_loss, val_acc, val_auc= validate(model, val_loader, criterion, device)

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Logging
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")

        # Log to wandb
        # run.log({
        #     "epoch": epoch + 1,
        #     "train_loss": train_loss,
        #     "train_acc": train_acc,
        #     "val_loss": val_loss,
        #     "val_acc": val_acc,
        #     "val_auc": val_auc
        # })

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
            #run.log({"best_val_auc": best_val_auc, "best_val_acc": best_val_acc})


    print(f"\n{phase_name} completed!")
    print(f"Best Val Acc: {best_val_acc:.2f}%, Best Val AUC: {best_val_auc:.4f}")

    return history

# FAZA 1

weight = torch.tensor([1.1]).to(Config.DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
optimizer = optim.AdamW(model.parameters(), Config.PHASE1_LR, weight_decay=0.5)

# 4. Scheduler - usunęliśmy 'verbose', bo w nowym PyTorch wywala błąd
scheduler_phase1 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',  # 'min' jeśli monitorujemy loss, 'max' jeśli AUC
    factor=0.5,
    patience=3
)
# 5. Uruchomienie fazy 1
history_phase1 = train_phase(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler_phase1,
    device=Config.DEVICE,
    num_epochs=Config.PHASE1_EPOCHS,
    phase_name="Phase 1: Training Classifier Head",
    save_path=os.path.join(Config.SAVE_DIR, "vit_phase1_best.pth")
)

