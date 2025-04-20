import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm  # Progress bar
from torchvision.models import EfficientNet_B1_Weights


# Check for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = "C:\\Users\\dtirt\\Desktop\\Folder\\Deepfake Detection Project\\deepfake_detection"
DATASET_PATH = os.path.join(BASE_DIR, "split_dataset")
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "tmp_checkpoint")

# Ensure checkpoint folder exists
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Model parameters
INPUT_SIZE = 240  # EfficientNetB1 expects 240x240
BATCH_SIZE = 16
NUM_EPOCHS = 20
MAX_MODELS = 5  # Keep only the last 5 models
LEARNING_RATE = 1e-4
torch.backends.cudnn.benchmark = True
# Data Augmentation
# Data Augmentation
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.4, 1.0)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(40),  
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.3),  
    transforms.RandomAffine(degrees=30, shear=20, translate=(0.1, 0.1)),  
    transforms.ToTensor(),  # Convert to Tensor BEFORE RandomErasing
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # Now it works!
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


transform_test = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Datasets
train_dataset = ImageFolder(root=TRAIN_PATH, transform=transform_train)
val_dataset = ImageFolder(root=VAL_PATH, transform=transform_test)
test_dataset = ImageFolder(root=TEST_PATH, transform=transform_test)

# Create Data Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# Define the Model (EfficientNetB1)
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepfakeDetector, self).__init__()
        self.model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        # Unfreeze last few layers for fine-tuning
        for param in list(self.model.features.parameters())[:100]:  # Freeze first 100 layers
            param.requires_grad = False
        
        # Replace classifier
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)  # Returns raw logits

# Initialize Model
model = DeepfakeDetector().to(DEVICE)

# Define Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# Training Function
def train_model():
    EARLY_STOPPING_PATIENCE = 6
    early_stopping_counter = 0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(NUM_EPOCHS):
        print(f"\n Epoch {epoch+1}/{NUM_EPOCHS}")

        # Training Phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)  # Correct


            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)  # Convert shape from [batch, 1] to [batch] 
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation Phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(DEVICE)
                labels = labels.float().to(DEVICE)  # Correct
                outputs = model(images)
                outputs = outputs.squeeze(1)  # Convert shape from [batch, 1] to [batch] 
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

        scheduler.step()

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_path = os.path.join(CHECKPOINT_PATH, f"best_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model: {best_model_path}")

            # Keep only last 5 models
            model_files = sorted(glob.glob(os.path.join(CHECKPOINT_PATH, "best_model_epoch_*.pt")), key=os.path.getmtime)
            if len(model_files) > MAX_MODELS:
                os.remove(model_files[0])

        else:
            early_stopping_counter += 1
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered. Stopping training!")
                break

    # Save Training History
    with open(os.path.join(BASE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Save Final Model as "best_model.pt"
    final_model_path = os.path.join(CHECKPOINT_PATH, "best_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved as: {final_model_path}")

    print("Training completed!")

# Run Training
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    train_model()