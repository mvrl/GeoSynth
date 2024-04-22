# Train GeoSynth

## 🧑‍💻 Setting up environment

Create a conda environment:

```python
conda env create -f environment.yaml
conda activate control
```
## 💬 Extracting LLaVA Captions

We use LLaVA-7b for automatically captioning each satellite image. Run the following command to get captions corresponding to each satellite image:

```python
python get_llava_captions.py
```

## 🛰️ Extracting SatCLIP Embeddings
Execute the following command to get all the geographic location embeddings:
```python
python get_satclip_embeds.py
```

## 🔥 Training

Setup all parameters of interest in `train.py`, then run:

```python
python train.py
```

