## Semi-supervised deep learning for fully convolutional networks<br><i>Official implementation of the MICCAI 2017 paper</i>

**Christoph Baur** (CAMP, TU Munich), **Shadi Albarqouni** (CAMP, TU Munich), **Nassir Navab** (CAMP, TU Munich and JHU, Baltimore)

**Abstract:**<br>
*Deep learning usually requires large amounts of labeled training data, but annotating data is costly and tedious. The framework of semi-supervised learning provides the means to use both labeled data and arbitrary amounts of unlabeled data for training. Recently, semi-supervised deep learning has been intensively studied for standard CNN architectures. However, Fully Convolutional Networks (FCNs) set the state-of-the-art for many image segmentation tasks. To the best of our knowledge, there is no existing semi-supervised learning method for such FCNs yet. We lift the concept of auxiliary manifold embedding for semi-supervised learning to FCNs with the help of Random Feature Embedding. In our experiments on the challenging task of MS Lesion Segmentation, we leverage the proposed framework for the purpose of domain adaptation and report substantial improvements over the baseline model.*

## Resources
* [Paper (MICCAI 2017)](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_36)
* [Paper (arXiv)](https://arxiv.org/abs/1703.06000)

## Requirements
* MATLAB 2017a (last tested with this version)
* SPM12 Toolbox
* MatConvNet beta 20 (already included here)
* MATLAB Image Processing Toolbox
* MATLAB Statistics Toolbox
* tSNE for MATLAB (if desired)

## Project/Folder Structure

* +CNN (namespace): holds helper and convenience functions for training a CNN using MatConvNet
* +dagnn (namespace): holds our custom layers and loss functions
* +IMDB (namespace): contains various methods to construct an image database (data, labels and dataset split information) used for training
* data: put your data here, if you want to
* extern: for your convenience already holds external dependencies, i.e. MatConvNet beta 20 and tSNE.
* models: save your model and training progress here
* test/CNN/MS: The original scripts for training and fine-tuning a U-Net as we have done in the paper
* util: holds helper functions and utilities
* NII.m: Class which represents a NII (Nifti) volume and provides convenience methods for visualization, extracting slices etc. Requires SPM12.
* Setup.m: Run this script before anything else to add dependencies to the MATLAB path

## How To

**PLEASE NOTE: This version has been tidied up and has not been tested for full functionality. Please open tickets if you face any troubles.**

0. Adjust any paths in Setup.m
1. Adjust any options and parameters in CNN_opts.
2. Generate an IMDB (MATLAB struct which holds data, labels and dataset split information) using CNN_createIMDB
3. Run CNN_train in order to train a model in a supervised fashion
4. Run CNN_finetune in oder to load a pretrained model and fine-tune it using Random Feature Embedding

## Cite Us

If you use some of the code in your work, please cite our paper:

```
@inproceedings{baur2017semi,
  title={Semi-supervised deep learning for fully convolutional networks},
  author={Baur, Christoph and Albarqouni, Shadi and Navab, Nassir},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={311--319},
  year={2017},
  organization={Springer}
}
```