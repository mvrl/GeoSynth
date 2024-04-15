# GeoSynth

This repository is the official implementation of [GeoSynth](https://arxiv.org/abs/2404.06637) [CVPRW, EarthVision, 2024].
GeoSynth is a suite of models for synthesizing satellite images with global style and image-driven layout control.

**[GeoSynth: Contextually-Aware High-Resolution Satellite Image Synthesis](https://arxiv.org/abs/2404.06637)** 
</br>
[Srikumar Sastry*](https://sites.wustl.edu/srikumarsastry/),
[Subash Khanal](https://subash-khanal.github.io/),
[Aayush Dhakal](https://scholar.google.com/citations?user=KawjT_8AAAAJ&hl=en),
[Nathan Jacobs](https://jacobsn.github.io/)
(*Corresponding Author)

[![arXiv](https://img.shields.io/badge/arXiv-2404.06637-red?style=flat&label=arXiv)](https://arxiv.org/abs/2404.06637)
[![Project Page](https://img.shields.io/badge/Project-Website-green)]()
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces-yellow?style=flat&logo=hug)](https://huggingface.co/spaces/MVRL/GeoSynth)

Models available on 🤗 HuggingFace:

GeoSynth-OSM: [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow?style=flat&logo=hug
)](https://huggingface.co/MVRL/GeoSynth-OSM)

GeoSynth-SAM: [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow?style=flat&logo=hug
)](https://huggingface.co/MVRL/GeoSynth-SAM)

GeoSynth-Canny: [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow?style=flat&logo=hug
)](https://huggingface.co/MVRL/GeoSynth-Canny)

All model `ckpt` files available here - [Link](#model-zoo) 

## ⏭️ Next
- [ ] Update Gradio demo
- [ ] Release Location-Aware GeoSynth Models to 🤗 HuggingFace
- [x] Release PyTorch `ckpt` files for all models
- [x] Release GeoSynth Models to 🤗 HuggingFace

## 🐨 Model Zoo
Download GeoSynth models from the given links below:

|Control|Location|Download Url|
|----------|--------|----------|
|-|❌|[Link](https://huggingface.co/MVRL/GeoSynth/blob/main/sd-base-geosynth.ckpt)|
|OSM|❌|[Link](https://huggingface.co/MVRL/GeoSynth-OSM/blob/main/geosynth-osm-text.ckpt)|
|SAM|❌| [Link](https://huggingface.co/MVRL/GeoSynth-SAM)|
|Canny|❌| [Link](https://huggingface.co/MVRL/GeoSynth-Canny)|
|-|✅|[Link](https://wustl.box.com/s/o1ooaunhaym7v1qj3yzj3vof0lskxyha)|
|OSM|✅|[Link](https://wustl.box.com/s/fudo44eznjwejcp3vql14by20rqqayfy)|
|SAM|✅| [Link](https://wustl.box.com/s/xuezslrnjxyz1d1ngtzvnm5ck2il4nx8)|
|Canny|✅| [Link](https://wustl.box.com/s/c3nfbdmcigiogqskemyc4h5soveiya8n)|


## 📑 Citation

```bibtex
@inproceedings{sastry2024geosynth,
  title={GeoSynth: Contextually-Aware High-Resolution Satellite Image Synthesis},
  author={Sastry, Srikumar and Khanal, Subash and Dhakal, Aayush and Jacobs, Nathan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2024}
}
```

## 🔍 Additional Links
Check out our lab website for other interesting works on geospatial understanding and mapping;
* Multi-Modal Vision Research Lab (MVRL) - [Link](https://mvrl.cse.wustl.edu/)
* Related Works from MVRL - [Link](https://mvrl.cse.wustl.edu/publications/)