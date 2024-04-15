import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, prompt_path, location_embeds_path):
        self.data = json.load(open(prompt_path, 'rt'))
        self.loc_embeds = np.load(location_embeds_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        loc_ind = item['class']

        source = cv2.imread("../"+source_filename)
        target = cv2.imread("../"+target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, location=self.loc_embeds[loc_ind])