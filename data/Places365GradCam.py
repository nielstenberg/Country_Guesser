import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2


model = models.resnet50(num_classes=365)

model.load_state_dict(torch.load("resnet50_places365.pth",
                                 map_location="cpu", weights_only=True))
model.eval()

print("Model loaded")

df = pd.read_parquet("split_data.parquet")
test_df = df[df.split == "test"]
row = test_df.sample(1).iloc[0]
response = requests.get(row.img_url)
img = Image.open(BytesIO(response.content)).convert("RGB")

print(f"Random TEST image: {row.image_id} | Country: {row.country}")
print("Image format:", img.size)
img.show()

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

model.eval()
with torch.no_grad():
    outputs = model(img_t)
    probs = torch.softmax(outputs, dim=1)[0]
    top3 = torch.topk(probs, 3)

classes_file = "https://raw.githubusercontent.com" \
               "/csailvision/places365/master/categories_places365.txt"
classes = [line.strip().split(' ')[0][3:] for line
           in requests.get(classes_file).text.splitlines()]

print("\nTop 3 predictions:")
for score, idx in zip(top3.values, top3.indices):
    print(f"{classes[idx]}: {score.item():.3f}")

print("\nTrue country (from your data):", row.country)

grads, acts = [], []


def forward_hook(module, input, output):
    acts.append(output)


def backward_hook(module, grad_in, grad_out):
    grads.append(grad_out[0])


target_layer = model.layer4[2].conv3
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

outputs = model(img_t)
top_classes = top3.indices[:3]  # top 3 predicted classes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, class_idx in enumerate(top_classes):
    grads, acts = [], []

    def forward_hook(module, input, output):
        acts.append(output)

    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0])

    target_layer = model.layer4[2].conv3
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

    axes[i].imshow(overlay)
    axes[i].set_title(f"{classes[class_idx]}")
    axes[i].axis('off')

    f_handle.remove()
    b_handle.remove()

plt.tight_layout()
plt.show()
