import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file
import pandas as pd
from tqdm import tqdm

# Define the ResNet-18 architecture as required [cite: 35]
def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Load Data & Target Model ---
# Using CIFAR-100 normalization parameters provided [cite: 46, 47]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
all_x = torch.stack([dataset[i][0] for i in range(1000)]).to(device)

print("Loading target model weights...")
target_path = "/content/tml26_task2/target_model/weights.safetensors"
target_state_dict = load_file(target_path, device="cpu")
target_model = make_model()
target_model.load_state_dict(target_state_dict)
target_model.to(device).eval()

with torch.no_grad():
    target_logits = target_model(all_x)
    target_preds = torch.argmax(target_logits, dim=1)
target_fc_weights = target_model.fc.weight.detach().flatten()
del target_model

# --- 2. Evaluate 360 Suspect Models ---
subset_ids = list(range(360))
results = []

for model_id in tqdm(subset_ids, desc="Checking Suspects"):
    # Zero-padded filename matching your folder structure [cite: 62]
    checkpoint_path = f"/content/tml26_task2/suspect_models/suspect_{model_id:03d}.safetensors"
    
    if not os.path.exists(checkpoint_path):
        results.append(0.0)
        continue

    # Load suspect and calculate similarity [cite: 113, 114]
    state_dict = load_file(checkpoint_path, device="cpu")
    suspect_model = make_model().to(device)
    suspect_model.load_state_dict(state_dict)
    suspect_model.eval()
    
    # Weight Similarity (White-box) 
    suspect_fc_weights = suspect_model.fc.weight.detach().flatten()
    w_sim = F.cosine_similarity(target_fc_weights.unsqueeze(0), suspect_fc_weights.unsqueeze(0)).item()
    
    # Functional Similarity (Knockoff/Agreement) [cite: 111]
    with torch.no_grad():
        suspect_preds = torch.argmax(suspect_model(all_x), dim=1)
    f_sim = (target_preds == suspect_preds).float().mean().item()
    
    # Combined confidence score [cite: 16, 17]
    results.append((0.5 * w_sim) + (0.5 * f_sim))
    
    del suspect_model
    torch.cuda.empty_cache()

# --- 3. Save Submission ---
pd.DataFrame({"id": subset_ids, "score": results}).to_csv("submission.csv", index=None)
print("Submission saved to submission.csv")