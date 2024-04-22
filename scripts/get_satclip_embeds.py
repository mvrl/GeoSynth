import torch
import pandas as pd
import numpy as np
from load import get_satclip

df = pd.read_csv("input_data_list.csv")
device = "cpu"

locs = torch.tensor(np.array(df[["X", "Y"]])).cpu()

model = get_satclip(
    "satclip-vit16-l40.ckpt", device=device
).eval()  # Only loads location encoder by default

with torch.no_grad():
    emb = model(locs.double().to(device)).detach().cpu().numpy()

np.save("location_embeds.npy", emb)
