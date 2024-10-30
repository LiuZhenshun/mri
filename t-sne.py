import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from model.model import build_model
from datasets import ACDCDataset

# Function to extract features
def extract_features(model, train_dataloader, val_dataloader, device):
    features_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(1).permute(0,1,4,2,3)
            features = model.forward_features(inputs)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(1).permute(0,1,4,2,3)
            features = model.forward_features(inputs)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return np.concatenate(features_list), np.concatenate(labels_list)


model = build_model("vit_base", in_chans=1, num_classes=5)

state_dict = torch.load("/home/comp/zhenshun/projects/mri/output/vit_base/models/checkpoint-199/mp_rank_00_model_states.pt", map_location="cpu")["module"]
model.load_state_dict(state_dict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

dataset_train  = ACDCDataset('/home/comp/zhenshun/datasets/ACDC/database/training/')
dataset_val  = ACDCDataset('/home/comp/zhenshun/datasets/ACDC/database/testing/')

dataloader_train = DataLoader(dataset_train, batch_size=10, shuffle=False)
dataloader_val = DataLoader(dataset_val, batch_size=10, shuffle=False)

features, labels = extract_features(model, dataloader_train, dataloader_val, device)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features)

# Plot the t-SNE results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')

# Define custom legend labels
class_labels = ['DCM', 'HCM', 'MINF', 'NOR', 'RV']

# Add legend with custom labels
handles, _ = scatter.legend_elements()
legend1 = plt.legend(handles, class_labels, title="Classes")
plt.gca().add_artist(legend1)

plt.title('t-SNE plot in ACDC dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.savefig("sne.png")