import torch
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(num_classes=365)

checkpoint = torch.hub.load_state_dict_from_url(
    "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
    progress=True,
    map_location=DEVICE
)

state_dict = checkpoint["state_dict"]

new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(new_state_dict)
model.to(DEVICE)
model.eval()

torch.save(model.state_dict(), "model/resnet50_places365.pth")
print("Model saved as 'resnet50_places365.pth'")
