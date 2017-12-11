import argparse
import datetime
import os
import sys
import timeit
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk
import matplotlib.pyplot as plt
import mialab.evaluation.metric as mtrc

import mialab.utilities.pipeline_utilities as putil

import numpy as np
from tensorflow.python.platform import app

"""
_dice.py : Dice coefficient for comparing set similarity.
"""

import numpy as np


def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    Dice = (2. * intersection.sum() / (im1.sum() + im2.sum()))

    print(Dice)
    return Dice

window_min = 500
window_max = 500

##Load some images to compare the different similarity measures#########################################################
image1 = sitk.ReadImage("Cyrcle1.png", sitk.sitkUInt8)
image2 = sitk.ReadImage("Cyrcle2.png", sitk.sitkUInt8)
########################################################################################################################


img = image1
msk = image2

overlay_img = sitk.LabelMapContourOverlay(sitk.Cast(msk, sitk.sitkLabelUInt8), sitk.Cast(sitk.IntensityWindowing(
    img, windowMinimum=window_min, windowMaximum=window_max), sitk.sitkUInt8), opacity=2, contourThickness=[2, 2])
    #We assume the original slice is isotropic, otherwise the display would be distorted
plt.imshow(sitk.GetArrayViewFromImage(overlay_img))
plt.axis('off')
plt.show()

im1 = np.asarray(image1).astype(np.bool)
im2 = np.asarray(image2).astype(np.bool)
''''''
script_dir = os.path.dirname(sys.argv[0])
result_dir = os.path.normpath(os.path.join(script_dir, './mia-result'))
#os.makedirs(result_dir, exist_ok=True)
# initialize evaluator
evaluator = putil.init_evaluator(result_dir)

evaluator.add_label(0, 'Background')
evaluator.add_label(1, 'Structure')
#evaluator.add_metric(mtrc.get_all_metrics())
evaluator.add_metric(mtrc.VolumeSimilarity())
evaluator.add_metric(mtrc.AreaUnderCurve())
evaluator.add_metric(mtrc.JaccardCoefficient())
evaluator.add_metric(mtrc.Sensitivity())
evaluator.evaluate(im1, im2, 'Patient1')

#dice(image1, image2)
#d_hausd = directed_hausdorff(np.array(image1), np.array(image2))

#print(d_hausd)
