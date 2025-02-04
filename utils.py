import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert (
            inputs.size() == target.size()
        ), f"predict {inputs.size()} & target {target.size()} shape do not match"

        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred: np.array, gt: np.array) -> tuple:
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        return (dice, hd95, jaccard)

    if pred.sum() > 0 and gt.sum() == 0:
        return (1, 0, 1)

    return (0, 0, 0)


def test_single_volume(
    image,
    label,
    net,
    classes,
    patch_size=None,
    test_save_path=None,
    case=None,
    z_spacing=1,
    is_rgb=False,
):
    if patch_size is None:
        patch_size = [256, 256]

    image, label = (
        image.squeeze(0).cpu().detach().numpy(),
        label.squeeze(0).cpu().detach().numpy(),
    )
    if len(image.shape) == 3 and not is_rgb:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            im_slice = image[ind, :, :]
            x, y = im_slice.shape[0], im_slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                im_slice = zoom(
                    im_slice, (patch_size[0] / x, patch_size[1] / y), order=3
                )  # previous using 0
            input_im = (
                torch.from_numpy(im_slice).unsqueeze(0).unsqueeze(0).float().cuda()
            )
            net.eval()
            with torch.no_grad():
                outputs = net(input_im)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        if len(image.shape) == 2:
            input_im = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        elif len(image.shape) == 3:
            input_im = torch.from_numpy(image).unsqueeze(0).float().cuda()

        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input_im), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))

        prd_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + "/" + case + "_pred.nii.gz")
        img_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(img_itk, test_save_path + "/" + case + "_img.nii.gz")
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(lab_itk, test_save_path + "/" + case + "_gt.nii.gz")

    return metric_list


"""TO DO Layer-Wise Learning Rate
    1.  Layer-Wise Learning Rate in PyTorch
        Implementing discriminative learning rate across model layers https://kozodoi.me/blog/20220329/discriminative-lr
    2. Layer-wise learning rate decay. What values to use? https://www.kaggle.com/c/commonlitreadabilityprize/discussion/251761

    Later work: 
    How to determine how many layers of a transformer model to freeze when fine-tuning? https://www.reddit.com/r/LanguageTechnology/comments/10pi16y/how_to_determine_how_many_layers_of_a_transformer/
"""