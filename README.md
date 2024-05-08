Code for [Soft labelling for semantic segmentation: Bringing coherence to label down-sampling](https://arxiv.org/pdf/2302.13961), currently under review. 
## Installation 

* The code is tested with pytorch 1.3 and python 3.6
* You can use ./Dockerfile to build an image.


## Weights

* Create a directory where you can keep large files. Ideally, not in this directory.
```bash
  > mkdir <large_asset_dir>
```

* Update `__C.ASSETS_PATH` in `config.py` to point at that directory

  __C.ASSETS_PATH=<large_asset_dir>

* Download pretrained weights from Supplementary Material and put into `<large_asset_dir>/seg_weights`

## Download/Prepare Data

If using Cityscapes, download Cityscapes data, then update `config.py` to set the path:
```python
__C.DATASET.CITYSCAPES_DIR=<path_to_cityscapes>
```

If using Mapillary, download Mapillary data, then update `config.py` to set the path:
```python
__C.DATASET.MAPILLARY_DIR=<path_to_mapillary>
```


## Running the code

The instructions below make use of a tool called `runx`, which we find useful to help automate experiment running and summarization. For more information about this tool, please see [runx](https://github.com/NVIDIA/runx).
In general, you can either use the runx-style commandlines shown below. Or you can call `python train.py <args ...>` directly if you like.


### Run inference on Cityscapes


```bash
> python -m runx.runx scripts/eval_cityscapes.yml -i
```

### Dump images for Cityscapes

```bash
> python -m runx.runx scripts/dump_cityscapes.yml -i
```

This will dump network output and composited images from running evaluation with the Cityscapes validation set. 

### Run inference and dump images on a folder of images

Modify the scripts/dump_folder.yml to point to the image folder to run inference:
```yalm
eval_folder:<path_to_image_folder>
```
Run:
```bash
> python -m runx.runx scripts/dump_folder.yml -i
```
This will dump network output, composited images and attention maps from running evaluation with the Cityscapes validation set. 

## Train a model

Train cityscapes
```bash
> python -m runx.runx scripts/train_cityscapes.yml -i
```
The first time this command is run, a centroid file has to be built for the dataset. It'll take about 10 minutes. The centroid file is used during training to know how to sample from the dataset in a class-uniform way.

This training run should deliver a model that achieves ~84.4 IOU.

## Code based on
Baseline code comes from [NVIDIA semantic segmentation framework](https://github.com/NVIDIA/semantic-segmentation)

