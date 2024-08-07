import os
from samgeo import SamGeo
from tqdm import tqdm 
import glob
from pathlib import Path

from samgeo.fast_sam import SamGeo
sam = SamGeo(model="FastSAM-x.pt")

sat_images = glob.glob("sat_images/*.jpeg")

for j in tqdm(range(len(sat_images))):
    try:
        sam.set_image(sat_images[j])
        sam.everything_prompt()
        sam.save_masks(output=os.path.join("sam_masks", Path(sat_images[j]).stem+".jpeg"), unique=True, foreground=False)
    except Exception as e:
        print(e)
        continue
