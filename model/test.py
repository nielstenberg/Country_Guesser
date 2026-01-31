from io import BytesIO
import pandas as pd
import requests
import os
import hashlib
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassF1Score
from torchvision import models, transforms
from tqdm import tqdm
from data.preprocessing.preprocessing import preprocess, input_transform
import csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
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

def topk_accuracy(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
    with torch.no_grad():
        topk = logits.topk(k, dim=1).indices
        correct = topk.eq(y.unsqueeze(1))
        return correct.any(dim=1).float().mean().item()
    
def load_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(num_classes=365)

    state = torch.load("model/resnet50_places365.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def get_topk_indices(logits, k):
    return logits.topk(k, dim=1).indices.squeeze(0).tolist()

def save_qualitative_examples(
    model,
    dataset,
    index_to_country,
    device,
    save_dir="plots"
):
    os.makedirs(save_dir, exist_ok=True)

    found = {
        "top1": None,
        "top3": None,
        "top5": None,
        "fail": None,
    }

    model.eval()
    
    indices = torch.randperm(len(dataset)).tolist()

    with torch.no_grad():
        for i in tqdm(indices, desc="Finding qualitative examples"):
            if all(v is not None for v in found.values()):
                break

            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)
            y = int(y)

            logits = model(x)

            top1 = get_topk_indices(logits, 1)
            top3 = get_topk_indices(logits, 2)
            top5 = get_topk_indices(logits, 5)

            if y == top1[0] and found["top1"] is None:
                found["top1"] = i
            elif y in top3 and y not in top1 and found["top3"] is None:
                found["top3"] = i
            elif y in top5 and y not in top3 and found["top5"] is None:
                found["top5"] = i
            elif y not in top5 and found["fail"] is None:
                found["fail"] = i

    for tag, idx in found.items():
        if idx is None:
            continue

        row = dataset.df.iloc[idx]
        url = row["img_url"]
        country = row["country"]

        fname = f"{tag}_{country}.jpg".replace(" ", "_")
        fpath = os.path.join(save_dir, fname)

        cache_name = hashlib.md5(url.encode()).hexdigest() + ".jpg"
        cache_path = os.path.join(CACHE_DIR, cache_name)

        if os.path.exists(cache_path):
            Image.open(cache_path).save(fpath)
        else:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            Image.open(BytesIO(r.content)).save(fpath)


def main():
    ckpt = torch.load(f"model/resnet50_country_best_1000.pth")
    country_index = ckpt["country_index"]
    num_classes = len(country_index)

    model = load_resnet50(num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    loss_fn = nn.CrossEntropyLoss()

    df = pd.read_parquet("data/datasets/split_data_1000.parquet")
    test_df = df[df["split"] == "test"].dropna(subset=["img_url", "country"])
    
    test_loader = DataLoader(
        CountryDataset(test_df, country_index, transform=input_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=16,
    )
    
    f1_metric = MulticlassF1Score(
        num_classes=num_classes,
        average="macro"
    ).to(DEVICE)
    
    test_loss = 0.0
    test_top1 = 0.0
    test_top3 = 0.0
    test_top5 = 0.0
    test_n = 0
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating", leave=False):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            raw_logits = model(x)
            loss = loss_fn(raw_logits, y)
            
            current_batch = x.size(0)
            test_loss += loss.item() * current_batch
            test_top1 += topk_accuracy(raw_logits, y, k=1) * current_batch
            test_top3 += topk_accuracy(raw_logits, y, k=3) * current_batch
            test_top5 += topk_accuracy(raw_logits, y, k=5) * current_batch
            test_n += current_batch
            
            preds = torch.argmax(raw_logits, dim=1)
            f1_metric.update(preds, y)
    
    test_loss /= test_n
    test_top1 /= test_n
    test_top3 /= test_n
    test_top5 /= test_n
    
    test_f1 = f1_metric.compute().item()
    f1_metric.reset()
    
    print(f"TEST RESULTS: LOSS: {test_loss} | top-1: {test_top1} | top-3: {test_top3} | top-5 {test_top5} | F1: {test_f1}")
    
    index_to_country = {v: k for k, v in country_index.items()}
    test_dataset = CountryDataset(test_df, country_index, transform=input_transform)

    save_qualitative_examples(
        model=model,
        dataset=test_dataset,
        index_to_country=index_to_country,
        device=DEVICE,
        save_dir="plots"
    )



if __name__ == "__main__":
    main()