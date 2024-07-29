from matplotlib import pyplot as plt
import json, os, torch, cv2, random
import numpy as np
from PIL import Image

def plot(rows, cols, count, im, gt = None, title = "Original Image"):

    plt.subplot(rows, cols, count)
    plt.imshow(im.squeeze(0).float()) if gt else plt.imshow((im * 255).cpu().permute(1, 2, 0).numpy().astype("uint8") * 255)
    plt.axis("off"); plt.title(title)

    return count + 1

def visualize(ds, n_ims):

    plt.figure(figsize = (25, 20))
    rows = n_ims // 4; cols = n_ims // rows
    count = 1
    indices = [random.randint(0, len(ds) - 1) for _ in range(n_ims)]

    for idx, index in enumerate(indices):

        if count == n_ims + 1: break

        im, gt,_ = ds[index]

        # First Plot
        count = plot(rows, cols, count, im = im)

        # Second Plot
        count = plot(rows, cols, count, im = gt.squeeze(0), gt = True, title = "GT Mask")


class Plot():

    def __init__(self, res):

        self.res = res

        self.visualize(metric1 = "tr_iou", metric2 = "val_iou", label1 = "Train IoU",
                  label2 = "Validation IoU", title = "Mean Intersection Over Union Learning Curve", ylabel = "mIoU Score")

        self.visualize(metric1 = "tr_pa", metric2 = "val_pa", label1 = "Train PA",
                  label2 = "Validation PA", title = "Pixel Accuracy Learning Curve", ylabel = "PA Score")

        self.visualize(metric1 = "tr_loss", metric2 = "val_loss", label1 = "Train Loss",
                  label2 = "Validation Loss", title = "Loss Learning Curve", ylabel = "Loss Value")

    def plot(self, metric, label): plt.plot(self.res[metric], label = label)

    def decorate(self, ylabel, title): plt.title(title); plt.xlabel("Epochs"); plt.ylabel(ylabel); plt.legend(); plt.show()

    def visualize(self, metric1, metric2, label1, label2, title, ylabel):

        plt.figure(figsize=(10, 5))
        self.plot(metric1, label1); self.plot(metric2, label2)
        self.decorate(ylabel, title)


def plot_single_image(dl, model, device, index):
    """
    Plots a single image from the dataset along with its ground truth and predicted mask.

    Parameters:
    dl (DataLoader): The data loader for the dataset.
    model (nn.Module): The trained model for prediction.
    device (torch.device): The device to run the model on.
    idx (int): The index of the image to plot.
    """

    # Get the specified data
    for idx, data in enumerate(dl):
        if idx == index:
            im, gt, _ = data

            # Get predicted mask
            with torch.no_grad():
                pred = torch.argmax(model(im.to(device)), dim=1).cpu()
            break

    def crop_to_non_black(image):
        """ Crop the image to the bounding box of non-black pixels """
        non_black_pixels = np.argwhere(image > 0)
        top_left = non_black_pixels.min(axis=0)
        bottom_right = non_black_pixels.max(axis=0)
        cropped_image = image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
        return cropped_image, top_left, bottom_right

    def pad_to_center(cropped_image, original_shape):
        """ Pad the cropped image back to the original shape, centering the content """
        padded_image = np.zeros(original_shape, dtype=cropped_image.dtype)
        pad_top = (original_shape[0] - cropped_image.shape[0]) // 2
        pad_left = (original_shape[1] - cropped_image.shape[1]) // 2
        padded_image[pad_top:pad_top + cropped_image.shape[0], pad_left:pad_left + cropped_image.shape[1]] = cropped_image
        return padded_image

    im_np = im.squeeze(0).squeeze(0).numpy()
    gt_np = gt.squeeze(0).squeeze(0).numpy()
    pred_np = pred.squeeze(0).numpy()

    im_cropped, _, _ = crop_to_non_black(im_np)
    gt_cropped, _, _ = crop_to_non_black(gt_np)
    pred_cropped, _, _ = crop_to_non_black(pred_np)

    im_padded = pad_to_center(im_cropped, im_np.shape)
    gt_padded = pad_to_center(gt_cropped, gt_np.shape)
    pred_padded = pad_to_center(pred_cropped, pred_np.shape)

    # Plot the image, ground truth, and predicted mask
    plt.figure(figsize=(15, 5))

    plt.imshow(pred_padded, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.show()