import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import torch.nn.functional as F
import numpy as np
import cv2
import os
import shutil

if os.path.exists("outputs"):
    shutil.rmtree("outputs")
os.makedirs("outputs", exist_ok=True)

state = torch.load("model/resnet50_country_best.pth", map_location="cpu")

num_classes = len(state["index_country"])

model = models.resnet50(num_classes=num_classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(state["model_state"])

state = torch.load("model/resnet50_country_best.pth", map_location="cpu")
model.load_state_dict(state["model_state"])
model.eval()

classes = [state["index_country"][i] for i in range(num_classes)]

print("Model loaded with", num_classes, "countries.")

df = pd.read_parquet("data/datasets/split_data_1000.parquet")
test_df = df[df.split == "test"]
row = test_df.sample(1).iloc[0]

response = requests.get(row.img_url)
img = Image.open(BytesIO(response.content)).convert("RGB")

print(f"Random TEST image: {row.image_id} | Country: {row.country}")
print("Image format:", img.size)

img.save("outputs/random_test_image.jpg")
print("Saved original image to outputs/random_test_image.jpg")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
img_t = transform(img).unsqueeze(0)
print("Transformed image shape:", img_t.shape)

with torch.no_grad():
    outputs = model(img_t)
    probs = torch.softmax(outputs, dim=1)[0]
    top3 = torch.topk(probs, 3)

print("\nTop 3 predictions:")
for score, idx in zip(top3.values, top3.indices):
    print(f"{classes[idx]}: {score.item():.3f}")

print("\nTrue country:", row.country)

grads, acts = [], []

def forward_hook(module, input, output):
    acts.append(output)

def backward_hook(module, grad_in, grad_out):
    grads.append(grad_out[0])

target_layer = model.layer4[2].conv3
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


top_classes = top3.indices[:3]

for i, class_idx in enumerate(top_classes):
    grads, acts = [], []

    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)

    outputs = model(img_t)
    model.zero_grad()
    outputs[0, class_idx].backward()

    weights = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts[0]).sum(1)).squeeze().detach()
    cam /= cam.max()

    cam_resized = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                size=(img.height, img.width),
                                mode='bilinear', align_corners=False)
    cam_resized = cam_resized.squeeze().numpy()

    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = np.array(img) * 0.5 + heatmap_color * 0.5
    overlay = overlay.astype(np.uint8)

    safe_class_name = classes[class_idx].replace("/", "_").replace(" ", "_")
    output_path = f"outputs/gradcam_{i}_{safe_class_name}.jpg"

    Image.fromarray(overlay).save(output_path)
    print(f"Saved Grad-CAM to {output_path}")

    f_handle.remove()
    b_handle.remove()

print("\nOutputs are in the 'outputs' folder.")
