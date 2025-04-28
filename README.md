# 3DReg-Mamba

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/optron-better-medical-image-registration-via/medical-image-registration-on-oasis)](https://paperswithcode.com/sota/medical-image-registration-on-oasis?p=optron-better-medical-image-registration-via)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/optron-better-medical-image-registration-via/medical-image-registration-on-ixi)](https://paperswithcode.com/sota/medical-image-registration-on-ixi?p=optron-better-medical-image-registration-via)

A novel dual-stream framework, 3DReg-Mamba, based on the Mamba model, treats 3D registration as a state-space modeling (SSM) problem capturing global dependencies in linear complexity. The framework consists of two core modules: FeatMamba for extracting global structural features with long-range dependencies, and MatchMamba for matching features between pairs of images by recursively integrating the global contexts of the two images, thus avoiding computationally costly block-by-block matching. In addition, the MatchMamba module employs a multidirectional coding strategy (including axial, coronal and sagittal flattening) and a feature voting fusion mechanism to enhance robustness and generality.

## Overall Architecture

![](.\img\image01.png)

**Modules Architecture Diagram**

![](C:\Users\Administrator\Desktop\3DReg-Mamba_github\img\image18.png)

## Usage

This repository currently provides examples of implementing, training, and inferring with the core model code. It also includes guidance on running the code on sample datasets. 

### Environment

We implement the code on `python 3.10.13`, `pytorch 2.1.1`, and `simpleitk 2.4.0`.

In particular, install the following two packages to ensure that Mamba will run successfully:

`causal_conv1d==1.1.3.post1`, `mamba_ssm==1.1.1`

### Dataset

- LPBA40 ([link](https://resource.loni.usc.edu/resources/atlases-downloads/))
- IXI ([link](https://brain-development.org/ixi-dataset/))
- OASIS ([link](https://sites.wustl.edu/oasisbrains/%5D))
- AbdomenCT-CT ([link](https://cloud.imi.uni-luebeck.de/s/32WaSRaTnFk2JeT))

###　Train and Infer Command

Before running the commands, please ensure that the dataset has been correctly placed. Taking the example of running the sample code on the LPBA40 dataset, ensure that the LPBA40 dataset is placed under `../Dataset/LPBA40_delineation/`. This will ensure that the code can run directly without encountering any path-related errors. (Here, `./` refers to the directory path where `Train.py` and `Infer.py` are located.)

For Linux:

Train:

```
export CUDA_VISIBLE_DEVICES=0 (If needed)
python Train.py
```

Infer:

```
export CUDA_VISIBLE_DEVICES=0 (If needed)
python Infer.py
```

The settings for each parameter of the code are located in `./utils/config.py`

## Example Results

**Visual results obtained by applying each alignment method for registration on the IXI dataset**

![](C:\Users\Administrator\Desktop\3DReg-Mamba_github\img\image07.png)





**Box plots show the Dice scores on 54 anatomical structures on the LPBA40 dataset using CycleMorph, VoxelMorph, TransMorph, TransMatch, OFG, CGNet, and our proposed method 3DRegMamba**

![](C:\Users\Administrator\Desktop\3DReg-Mamba_github\img\image13.png)





**Qualitative comparison of intermediate hierarchical features extracted by CGNet and our proposed 3DReg-Mamba**



![](C:\Users\Administrator\Desktop\3DReg-Mamba_github\img\image08.png)



## Baseline Method

We compared 3DReg-Mamba with **eight** baseline registration methods . The links will take you to their official repositories.

- SyN/ANTsPy([Official Website](https://github.com/ANTsX/ANTsPy))
- NiftyReg([Official Website](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg))
- VoxelMorph([Official Website](https://github.com/voxelmorph/voxelmorph))
- CycleMorph([Official Website](https://github.com/boahK/MEDIA_CycleMorph))
- TransMorph([Official Website](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/tree/main?tab=readme-ov-file))
- TransMatch([Official Website](https://github.com/tzayuan/TransMatch_TMI))
- OFG([Official Website](https://github.com/cilix-ai/on-the-fly-guidance))
- CGNet([Official Website](https://github.com/scu1996cy/CGNet))

