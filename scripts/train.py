import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data import Dataset
from ..ControlNet.cldm.logger import ImageLogger
from ..ControlNet.cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import os


# Configs
resume_path = 'GeoSynth/scripts/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 2000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('GeoSynth/scripts/models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

checkpoint = ModelCheckpoint(
        dirpath=os.path.join("checkpoint", 'geosynth'),
        filename='model_geosynth',
        every_n_train_steps=150
)
# Misc
dataset = Dataset(prompt_path='GeoSynth/scripts/prompt_with_locations.json', location_embeds_path='GeoSynth/scripts/location_embeds.npy')
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True, persistent_workers=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(accelerator='gpu',
strategy='ddp',
devices=2,
precision=32,
max_epochs=42,
callbacks=[logger, checkpoint],
accumulate_grad_batches=16
)

# Train!
trainer.fit(model, dataloader)