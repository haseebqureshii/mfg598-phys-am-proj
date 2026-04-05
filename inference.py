import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- SETTINGS ---
MODEL_PATH = "meltpool_cnn_model.pth"
# Point this to a specific image you want to test
TEST_IMG = r"D:\ASU\Spring 26\AI in Additive Manufacturing\proj\mds2-2923\Micrographs\IN718_20210323_M290\Snap-121.tif"
EXCEL_PATH = r"D:\ASU\Spring 26\AI in Additive Manufacturing\proj\mds2-2923\Master_TrackList_Measurements.xlsx"

# --- THE FIX: IDENTICAL CLASS STRUCTURE ---
# This class must exactly match the one in your train_meltpool_cnn.py
class MeltPoolResNet(nn.Module):
    def __init__(self):
        super(MeltPoolResNet, self).__init__()
        # Initialize the same backbone structure
        self.backbone = models.resnet18()
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2) 
        )

    def forward(self, x):
        return self.backbone(x)

def run_inference():
    device = torch.device("cpu")
    
    # 1. Initialize Scaler 
    # (We must fit it on the original data so inverse_transform works correctly)
    if not os.path.exists(EXCEL_PATH):
        print(f"Error: Could not find Excel at {EXCEL_PATH}")
        return
        
    print("Fitting scaler on training data range...")
    df = pd.read_excel(EXCEL_PATH, sheet_name='Data')
    df = df[['Width (µm)', 'Depth (µm)']].dropna()
    scaler = StandardScaler()
    scaler.fit(df.values)

    # 2. Load Model
    print(f"Loading weights from {MODEL_PATH}...")
    model = MeltPoolResNet()
    try:
        # This will now succeed because the class has the 'backbone' attribute
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(TEST_IMG):
        print(f"Error: Test image not found at {TEST_IMG}")
        return

    image = Image.open(TEST_IMG).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    # 4. Run Prediction
    with torch.no_grad():
        output = model(input_tensor)
        # Convert normalized predictions back to real Microns
        predictions = scaler.inverse_transform(output.numpy())
        
    width, depth = predictions[0]
    
    print(f"\n" + "="*40)
    print(f"   VIRTUAL METROLOGY PREDICTION")
    print(f"="*40)
    print(f"File:  {os.path.basename(TEST_IMG)}")
    print(f"Width: {width:.2f} µm")
    print(f"Depth: {depth:.2f} µm")
    print(f"="*40)

if __name__ == "__main__":
    run_inference()