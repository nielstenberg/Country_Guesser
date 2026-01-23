from io import BytesIO
import pandas as pd
import requests
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3
EPOCHS = 5

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

        r = self.session.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")

        x = self.transform(img)
        y = self.country_index[country]

        return x, y

def load_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(num_classes=365)

    state = torch.load("resnet50_places365.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()

def train():
    df = pd.read_parquet("split_data.parquet")
    
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
        batch_size = 16,
        shuffle=True,
        num_workers = 0
    )

    val_loader = DataLoader(
        CountryDataset(val_df, country_index, tf),
        batch_size = 16,
        shuffle=False,
        num_workers = 0
    )

    model = load_resnet50(num_classes).to(DEVICE)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr = LR)
    best_validation = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n = 0

        for x, y in tqdm(train_loader, desc =f"Epoch {epoch}/{EPOCHS} [train]"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            raw_logits = model(x)
            loss = loss_fn(raw_logits, y)
            loss.backward()
            optimizer.step()

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
            for x, y in tqdm(val_loader, desc =f"Epoch {epoch}/{EPOCHS} [validation]"):
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

            tqdm.write(f"Epoch {epoch}/{EPOCHS} | "
                f"Train loss {train_loss:.4f} accuracy {train_acc:.4f} | "
                f"Validation loss {validation_loss:.4f} accuracy {validation_acc:.4f}")

        if validation_acc > best_validation:
            best_validation = validation_acc
            torch.save({
                "model_state": model.state_dict(),
                "country_index": country_index,
                "index_country": index_country},
                "resnet50_country_best.pth")
        
    tqdm.write(f"Best validation acccuracy: {best_validation:.4f}")

if __name__ == "__main__":
    train()
