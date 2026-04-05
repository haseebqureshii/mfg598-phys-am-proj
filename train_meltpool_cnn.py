import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BASE_PATH = r"D:\ASU\Spring 26\AI in Additive Manufacturing\proj\mds2-2923"
EXCEL_PATH = os.path.join(BASE_PATH, "Master_TrackList_Measurements.xlsx")
IMAGE_DIR = os.path.join(BASE_PATH, "Micrographs")
SAVE_PATH = "meltpool_cnn_model.pth"

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 30
IMG_SIZE = 224 # Standard for ResNet

# --- DATASET CLASS ---
class NISTMeltPoolDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        folder = str(self.df.iloc[idx]['Folder Name']).strip()
        img_base_name = str(self.df.iloc[idx]['Image Name']).strip()
        folder_path = os.path.join(self.root_dir, folder)
        
        # Robust Search: Handle .tif vs .tiff and filter out annotated 'measurement' files
        found_path = None
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                # Match file starting with the Image Name (case-insensitive)
                if f.lower().startswith(img_base_name.lower()):
                    # Prioritize raw images by avoiding those with measurement labels in the name
                    if "measure" not in f.lower() and "box" not in f.lower():
                        found_path = os.path.join(folder_path, f)
                        break
                    found_path = os.path.join(folder_path, f)

        # Fallback if no file was found
        if found_path is None or not os.path.exists(found_path):
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            # print(f"Warning: No valid file found for {img_base_name} in {folder}") # Optional debug
        else:
            image = Image.open(found_path).convert('RGB')

        # Targets: Width and Depth
        targets = self.df.iloc[idx][['Width (µm)', 'Depth (µm)']].values.astype('float32')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(targets)

# --- MODEL ARCHITECTURE ---
class MeltPoolResNet(nn.Module):
    def __init__(self):
        super(MeltPoolResNet, self).__init__()
        # Use modern weights parameter to avoid warnings and get latest ImageNet weights
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # NOTE: For old CPUs, you could freeze the backbone layers to speed up training:
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
            
        # Replace the last fully connected layer for multi-output regression (2 outputs)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2) 
        )

    def forward(self, x):
        return self.backbone(x)

# --- TRAINING LOOP ---
def train_model():
    # 1. Load and Clean Metadata from 'Data' sheet
    if not os.path.exists(EXCEL_PATH):
        print(f"Error: Could not find Excel file at {EXCEL_PATH}")
        return

    df = pd.read_excel(EXCEL_PATH, sheet_name='Data')
    df = df[['Folder Name', 'Image Name', 'Width (µm)', 'Depth (µm)']].dropna()
    
    # 2. Scale Targets (Critical for multi-output regression convergence)
    scaler = StandardScaler()
    df[['Width (µm)', 'Depth (µm)']] = scaler.fit_transform(df[['Width (µm)', 'Depth (µm)']])

    # 3. Split Data (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 4. Standard ResNet Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5. DataLoaders
    train_set = NISTMeltPoolDataset(train_df, IMAGE_DIR, transform=train_transform)
    val_set = NISTMeltPoolDataset(val_df, IMAGE_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Initialize Model, Loss (MSE for regression), Optimizer (Adam)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeltPoolResNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. Execution
    print(f"Starting training on {device}...")
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * images.size(0)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_epoch_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train MSE: {epoch_loss:.4f} | Val MSE: {val_epoch_loss:.4f}")

    # 8. Save Weights
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved successfully to {SAVE_PATH}")

    # 9. Plot Results
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Melt Pool Geometry Regression Training History')
    plt.legend()
    plt.savefig('learning_curve.png')
    plt.show()

if __name__ == "__main__":
    train_model()