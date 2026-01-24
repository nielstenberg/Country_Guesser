from io import BytesIO
import pandas as pd
import requests
import os
import hashlib
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3
EPOCHS = 5

PHASES = [
    {
        "name": "linear_probe",
        "epochs": 10,
        "unfreeze": ["fc"],
        "lrs": {"fc": 1e-3},
    },
    {
        "name": "finetune_layer4",
        "epochs": 8,
        "unfreeze": ["layer4", "fc"],
        "lrs": {"layer4": 1e-4, "fc": 1e-3},
    },
]

CACHE_DIR = "/tmp/country_images"
os.makedirs(CACHE_DIR, exist_ok=True)

class CountryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, country_index: dict, transform):
        self.df = df.reset_index(drop=True)
        self.country_index = country_index
        self.transform = transform
        self.session = requests.Session()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        url = row["img_url"]
        country = row["country"]
        
        fname = hashlib.md5(url.encode()).hexdigest() + ".jpg"
        fpath = os.path.join(CACHE_DIR, fname)

        if not os.path.exists(fpath):
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            with open(fpath, "wb") as f:
                f.write(r.content)
        
        img = Image.open(fpath).convert("RGB")

        x = self.transform(img)
        y = self.country_index[country]

        return x, y
    
def configure_trainable_params(model, unfreeze):
    for p in model.parameters():
        p.requires_grad = False

    if "all" in unfreeze:
        for p in model.parameters():
            p.requires_grad = True
        return

    for name in unfreeze:
        module = getattr(model, name)
        for p in module.parameters():
            p.requires_grad = True
            
def make_optimizer(model, lrs):
    param_groups = []

    if "all" in lrs:
        return torch.optim.Adam(model.parameters(), lr=lrs["all"])

    for name, lr in lrs.items():
        module = getattr(model, name)
        param_groups.append({"params": module.parameters(), "lr": lr})

    return torch.optim.Adam(param_groups)

def load_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(num_classes=365)

    state = torch.load("saved_models/resnet50_places365.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()

def train():
    df = pd.read_parquet("../data/datasets/split_data.parquet")
    
    required_columns = {"img_url", "country", "split"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"split_data.parquet missing columns: {missing_columns}")
    
    train_df = df[df["split"] == "train"].dropna(subset=["img_url", "country"])
    val_df = df[df["split"] == "val"].dropna(subset=["img_url", "country"])
    if len(val_df) == 0:
        val_df = df[df["split"] == "test"].dropna(subset=["img_url", "country"])

    country_list = sorted(train_df["country"].unique())
    country_index = {country: index for index, country in enumerate(country_list)}
    index_country = {index: country for country, index in country_index.items()}
    num_classes = len(country_list)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_loader = DataLoader(
        CountryDataset(train_df, country_index, tf),
        batch_size = 64,
        shuffle=True,
        num_workers = 16,
        pin_memory=True
    )

    val_loader = DataLoader(
        CountryDataset(val_df, country_index, tf),
        batch_size = 64,
        shuffle=False,
        num_workers = 16,
        pin_memory=True
    )

    model = load_resnet50(num_classes).to(DEVICE)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True

    loss_fn = nn.CrossEntropyLoss()
    best_validation = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    global_epoch = 0
    
    for phase in PHASES:
        print(f"\n=== Phase: {phase['name']} ===")
        configure_trainable_params(model, phase["unfreeze"])
        optimizer = make_optimizer(model, phase["lrs"])

        for epoch in range(1, phase["epochs"] + 1):
            global_epoch += 1
            model.train()
            
            train_loss = 0.0
            train_acc = 0.0
            n = 0

            for x, y in tqdm(train_loader, desc =f"Epoch {global_epoch} [{phase['name']}] {epoch}/{phase['epochs']} [train]"):
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    raw_logits = model(x)
                    loss = loss_fn(raw_logits, y)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()


                current_batch = x.size(0)
                train_loss += loss.item() * current_batch
                train_acc += accuracy(raw_logits, y) * current_batch
                n += current_batch

            train_loss /= n
            train_acc /= n

            model.eval()
            validation_loss = 0.0
            validation_acc = 0.0
            validation_n = 0
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc =f"Epoch {global_epoch} [{phase['name']}] {epoch}/{phase['epochs']} [validation]"):
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    
                    raw_logits = model(x)
                    loss = loss_fn(raw_logits, y)

                    current_batch = x.size(0)
                    validation_loss += loss.item() * current_batch
                    validation_acc += accuracy(raw_logits, y) * current_batch
                    validation_n += current_batch

                validation_loss /= validation_n
                validation_acc /= validation_n

                tqdm.write(f"Epoch {global_epoch} [{phase['name']}] {epoch}/{phase['epochs']} | "
                    f"Train loss {train_loss:.4f} accuracy {train_acc:.4f} | "
                    f"Validation loss {validation_loss:.4f} accuracy {validation_acc:.4f}")

            if validation_acc > best_validation:
                best_validation = validation_acc
                torch.save({
                    "model_state": model.state_dict(),
                    "country_index": country_index,
                    "index_country": index_country},
                    "saved_models/resnet50_country_best.pth")
        
    tqdm.write(f"Best validation acccuracy: {best_validation:.4f}")

if __name__ == "__main__":
    train()
