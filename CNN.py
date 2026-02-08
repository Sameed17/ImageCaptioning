import os
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

IMAGE_DIR = "data/Images"
OUTPUT_FILE = "flickr30k_features.pkl"

class FlickrDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".jpeg"))]
        self.transform = transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, name)
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), name


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])
    model = nn.DataParallel(model).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = FlickrDataset(IMAGE_DIR, transform)
    loader = DataLoader(dataset, batch_size=128, num_workers=0)

    features_dict = {}
    with torch.no_grad():
        for imgs, names in tqdm(loader, desc="Extracting Features"):
            feats = model(imgs.to(device)).view(imgs.size(0), -1)
            for i, name in enumerate(names):
                features_dict[name] = feats[i].cpu().numpy()

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(features_dict, f)

    print(f"Success! {len(features_dict)} images processed and saved to {OUTPUT_FILE}")
