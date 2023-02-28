import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class ProstateMRDataset(torch.utils.data.Dataset):
    def __init__(self, paths, img_size):
        """
        Constructor

        Parameters
        ----------
        paths : paths to patient data
        img_size : size images to be interpolated to
        """

        self.mr_image_list = []
        self.mask_list = []
        # load images
        for path in paths:
            self.mr_image_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe.mhd"))
            )
            self.mask_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd"))
            )

        # number of patients and slices in the dataset
        self.no_patients = len(self.mr_image_list)
        self.no_slices = self.mr_image_list[0].shape[0]

        # transforms to resize images
        self.img_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(256),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )
        # standardise intensities based on mean and std deviation
        self.train_data_mean = np.mean(self.mr_image_list)
        self.train_data_std = np.std(self.mr_image_list)
        self.norm_transform = transforms.Normalize(
            self.train_data_mean, self.train_data_std
        )

    def __len__(self):
        """
        Returns length of dataset
        """
        return self.no_patients * self.no_slices

    def __getitem__(self, index):
        """
        Returns the preprocessing MR image and corresponding segementation
        for a given index
        """

        # compute which slice an index corresponds to
        patient = index // self.no_slices
        the_slice = index - (patient * self.no_slices)

        return (
            self.norm_transform(
                self.img_transform(
                    self.mr_image_list[patient][the_slice, ...].astype(np.float32)
                )
            ),
            self.img_transform(
                (self.mask_list[patient][the_slice, ...] > 0).astype(np.int32)
            ),
        )


class DiceBCELoss(nn.Module):
    """
    Class for the loss function:
    sum of Dice score and binary cross-entropy
    """

    def __init__(self, weight=None, size_average=True):
        # Constructor
        super(DiceBCELoss, self).__init__()

    def forward(self, outputs, targets, smooth=1):
        """
        Calculates segmentation loss for training

        Parameters
        ----------
        outputs : predictions of segmentation mode
        targets : ground-truth labels
        smooth : smooth parameter for dice score
            avoids division by zero, by default 1

        Returns
        -------
        The sum of the dice loss and binary cross-entropy
        """
        outputs = torch.sigmoid(outputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # compute Dice
        intersection = (inputs * targets).sum()
        dice_loss = # TODO
        BCE = # TODO

        return BCE + dice_loss
