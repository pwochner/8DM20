**Additional Project Description**

An alternative approach for automatic segmentation, which does not require registration, is via pixel-based classification. Here the delineated patient images form the training set, where each pixel is a sample, labeled as foreground (organ of interest) or background. The training set is used to train a classifier, with a common approach being to use a deep learning model such as a U-Net, demonstrated in Figure 1.

![figure 1](segment.png "Title")
*Figure 1: a schematic of a U-Net model, taking the prostate MR image as input and outputting the binary segmentation mask.
*

