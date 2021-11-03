# Few-shot Semantic Segmentation with Self-supervision from Pseudo-classes

This is the implementation of paper 
[Few-shot Semantic Segmentation with Self-supervision from Pseudo-classes](https://arxiv.org/abs/2110.11742)
that has been accepted to BMVC 2011.

This project is built upon [this](https://github.com/dvlab-research/PFENet) repository.

# Get Started

## Datasets and Data Preparation

### PASCAL-5i

PASCAL-5i is based on the [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and 
[SBD](http://home.bharathh.info/pubs/codes/SBD/download.html). Prepare PASCAL-5i data by:

1. Download [VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and
[SegmentationClassAug](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp),
put them under `./data/pascal`
2. Run `python ./data/prepare_pascal.py`

Validation set includes VOC validation images.
Training set includes VOC training images and part of SBD training images 
(from this [list](http://home.bharathh.info/pubs/codes/SBD/download.html))
which do not overlap with the validation set.
