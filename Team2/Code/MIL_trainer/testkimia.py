# %%
from models import create_model
import torch

model = create_model(pretrained=True)
print(next(model.features.parameters()).requires_grad)
print(torch.mean(next(model.features.parameters())).item())