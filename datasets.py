import torch
from torch.utils.data import Subset
import SimpleITK as sitk
from torch.utils.data import random_split, Dataset, DataLoader
import glob
from PIL import Image
import numpy as np

### Segmentation over the full femur ###
class CustomSegmentationDataset(Dataset):

    def __init__(self, root, transformations=None):
        im_nrrd_paths = sorted(glob.glob(f"{root}/Images/*.nrrd"))
        gt_nrrd_paths = sorted(glob.glob(f"{root}/GoldStandard/*.nrrd"))

        numeri_lista = []

        for percorso in gt_nrrd_paths:
            # Estrai il numero alla fine del percorso del file vicino alla 'L'
            numero = int(percorso.split('/')[-1].split('_')[0][1:])
            numeri_lista.append(numero)


        indice_da_rimuovere = numeri_lista.index(442)

        # Rimuovi l'elemento corrispondente all'indice trovato da gt_paths
        elemento_da_rimuovere = gt_nrrd_paths[indice_da_rimuovere]
        gt_nrrd_paths.remove(elemento_da_rimuovere)

        # images_path.append(im_nrrd_paths)
        # gt_paths.append(gt_nrrd_paths)
        self.ims, self.gts, self.orientation = self.get_slices(im_nrrd_paths, gt_nrrd_paths)
        self.transformations = transformations
        self.n_cls = 2

        assert len(self.ims) == len(self.gts) == len(self.orientation)

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im, gt, orientation = self.ims[idx], self.gts[idx], self.orientation[idx]
        if self.transformations:
            im, gt = self.apply_transformations(im, gt)

        # For visualization purposes
        im = self.preprocess_im(im)
        # For the cases when label equals to 2; to avoid CE Loss error
        gt[gt > 1] = 1

        return im.float(), gt.unsqueeze(0).long(), orientation

    def apply_transformations(self, im, gt): transformed = self.transformations(image = im, mask = gt); return transformed["image"], transformed["mask"]

    def get_slices(self, im_nrrd_paths, gt_nrrd_paths):
        ims, gts, orientation = [], [], []

        for index, (im_nrrd, gt_nrrd) in enumerate(zip(im_nrrd_paths, gt_nrrd_paths)):
            print(f"NRRD file number {index + 1} is being converted...")
            nrrd_im_data, nrrd_gt_data = self.read_nrrd(im_nrrd, gt_nrrd)

            # Slicing in axial, coronal, and sagittal directions
            for axis in range(3):
                for idx in range(nrrd_im_data.shape[axis]):
                    if axis == 0:
                        im_slice = nrrd_im_data[idx, :, :]
                        gt_slice = nrrd_gt_data[idx, :, :]
                        orient = 0
                    elif axis == 1:
                        im_slice = nrrd_im_data[:, idx, :]
                        gt_slice = nrrd_gt_data[:, idx, :]
                        orient = 1
                    elif axis == 2:
                        im_slice = nrrd_im_data[:, :, idx]
                        gt_slice = nrrd_gt_data[:, :, idx]
                        orient = 2

                    if len(np.unique(gt_slice)) == 2:
                        # Resize image and mask to 256x256
                        im_resized = sitk.GetArrayFromImage(sitk.Resample(sitk.GetImageFromArray(im_slice), (256, 256)))
                        gt_resized = sitk.GetArrayFromImage(sitk.Resample(sitk.GetImageFromArray(gt_slice), (256, 256)))
                        ims.append(im_resized)
                        gts.append(gt_resized)
                        orientation.append((index, orient))

        return ims, gts, orientation

    def read_nrrd(self, im_path, gt_path):
        im_data = sitk.ReadImage(im_path)
        gt_data = sitk.ReadImage(gt_path)
        return sitk.GetArrayFromImage(im_data), sitk.GetArrayFromImage(gt_data)

    def preprocess_im(self, im):
        # Push background value of -3024 to -1000
        im[im == -3024] = -1000

        # Min-max scaling
        min_val = torch.min(im)
        max_val = torch.max(im)

        # Ensure the min value is not equal to the max value to avoid division by zero
        if min_val != max_val:
            im = (im - min_val) / (max_val - min_val)
        else:
            im = torch.zeros_like(im)

        return im
# Example usage to plot an image before DataLoader creation
def get_dlsfull(root, transformations, bs, split = [0.6, 0.2, 0.2], ns = 4):

    #assert sum(split) == 1., "Sum of the split must be exactly 1"

    ds = CustomSegmentationDataset(root = root, transformations = transformations)
    n_cls = ds.n_cls

    tr_len = int(len(ds) * split[0])
    val_len = int(len(ds) * split[1])
    test_len = len(ds) - (tr_len + val_len)

    ds_len = len(ds)
    indices = list(range(ds_len))

    # Define the lengths for the splits
    tr_len = int(0.6 * ds_len)  # 60% for training
    val_len = int(0.2 * ds_len)  # 20% for validation
    test_len = ds_len - tr_len - val_len  # Remaining 20% for testing

    # Split indices
    tr_indices = indices[:tr_len]
    val_indices = indices[tr_len:tr_len + val_len]
    test_indices = indices[tr_len + val_len:]

    # Create subsets
    tr_ds = Subset(ds, tr_indices)
    val_ds = Subset(ds, val_indices)
    test_ds = Subset(ds, test_indices)

    print(f"\nThere are {len(tr_ds)} number of images in the train set")
    print(f"There are {len(val_ds)} number of images in the validation set")
    print(f"There are {len(test_ds)} number of images in the test set\n")
    # Get dataloaders
    tr_dl  = DataLoader(dataset = tr_ds, batch_size = bs, shuffle = True, num_workers = ns)
    val_dl = DataLoader(dataset = val_ds, batch_size = bs, shuffle = False, num_workers = ns)
    test_dl = DataLoader(dataset = test_ds, batch_size = 1, shuffle = False, num_workers = ns)

    return ds, tr_dl, val_dl, test_dl, n_cls

### Segmentation over the heads ###
class CriticalFemourSegmentation(Dataset):

    def __init__(self, root, transformations=None):
        im_nrrd_paths = sorted(glob.glob(f"{root}/Images/*.nrrd"))
        gt_nrrd_paths = sorted(glob.glob(f"{root}/GoldStandard/*.nrrd"))

        numeri_lista = []

        for percorso in gt_nrrd_paths:
            # Estrai il numero alla fine del percorso del file vicino alla 'L'
            numero = int(percorso.split('/')[-1].split('_')[0][1:])
            numeri_lista.append(numero)


        indice_da_rimuovere = numeri_lista.index(442)

        # Rimuovi l'elemento corrispondente all'indice trovato da gt_paths
        elemento_da_rimuovere = gt_nrrd_paths[indice_da_rimuovere]
        gt_nrrd_paths.remove(elemento_da_rimuovere)

        self.ims, self.gts, self.slice_indices = self.get_slices(im_nrrd_paths, gt_nrrd_paths)
        self.transformations = transformations
        self.n_cls = 2

        assert len(self.ims) == len(self.gts)

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im, gt = self.ims[idx], self.gts[idx]
        slice_index = self.slice_indices[idx]
        if self.transformations:
            im, gt = self.apply_transformations(im, gt)

        # For visualization purposes
        im = self.preprocess_im(im)
        # For the cases when label equals to 2; to avoid CE Loss error
        gt[gt > 1] = 1

        return im.float(), gt.unsqueeze(0).long(), slice_index

    def apply_transformations(self, im, gt):
        transformed = self.transformations(image=im, mask=gt)
        return transformed["image"], transformed["mask"]

    def get_slices(self, im_nrrd_paths, gt_nrrd_paths):
        ims, gts, slice_indices = [], [], []

        for index, (im_nrrd, gt_nrrd) in enumerate(zip(im_nrrd_paths, gt_nrrd_paths)):
            print(f"NRRD file number {index + 1} is being converted...")
            nrrd_im_data, nrrd_gt_data = self.read_nrrd(im_nrrd, gt_nrrd)
            # Collect axial slices (axis 0)
            axial_slices = []
            for idx in range(nrrd_im_data.shape[0]):
                im_slice = nrrd_im_data[idx, :, :]
                gt_slice = nrrd_gt_data[idx, :, :]
                if len(np.unique(gt_slice)) == 2 and not np.all(im_slice == 0):
                    axial_slices.append((im_slice, gt_slice, index))

            # Keep only the first 20 and last 20 slices that are not full black
            axial_slices = axial_slices[:20] + axial_slices[-20:]

            for im_slice, gt_slice, slice_index in axial_slices:
                # Resize image and mask to 256x256
                im_resized = sitk.GetArrayFromImage(sitk.Resample(sitk.GetImageFromArray(im_slice), (256, 256)))
                gt_resized = sitk.GetArrayFromImage(sitk.Resample(sitk.GetImageFromArray(gt_slice), (256, 256)))
                ims.append(im_resized)
                gts.append(gt_resized)
                slice_indices.append(slice_index)

        return ims, gts, slice_indices

    def read_nrrd(self, im_path, gt_path):
        im_data = sitk.ReadImage(im_path)
        gt_data = sitk.ReadImage(gt_path)
        return sitk.GetArrayFromImage(im_data), sitk.GetArrayFromImage(gt_data)

    def preprocess_im(self, im):
        # Push background value of -3024 to -1000
        im[im == -3024] = -1000

        # Min-max scaling
        min_val = torch.min(im)
        max_val = torch.max(im)

        # Ensure the min value is not equal to the max value to avoid division by zero
        if min_val != max_val:
            im = (im - min_val) / (max_val - min_val)
        else:
            im = torch.zeros_like(im)

        return im

# Example usage to plot an image before DataLoader creation
def get_dls(root, transformations, bs, split=[0.6, 0.2, 0.2], ns=4):
    ds = CriticalFemourSegmentation(root=root, transformations=transformations)
    n_cls = ds.n_cls

    tr_len = int(len(ds) * split[0])
    val_len = int(len(ds) * split[1])
    test_len = len(ds) - (tr_len + val_len)

    ds_len = len(ds)
    indices = list(range(ds_len))

    # Define the lengths for the splits
    tr_len = int(0.6 * ds_len)  # 60% for training
    val_len = int(0.2 * ds_len)  # 20% for validation

    # Split indices
    tr_indices = indices[:tr_len]
    val_indices = indices[tr_len:tr_len + val_len]
    test_indices = indices[tr_len + val_len:]

    # Create subsets
    tr_ds = Subset(ds, tr_indices)
    val_ds = Subset(ds, val_indices)
    test_ds = Subset(ds, test_indices)


    print(f"\nThere are {len(tr_ds)} number of images in the train set")
    print(f"There are {len(val_ds)} number of images in the validation set")
    print(f"There are {len(test_ds)} number of images in the test set\n")

    # Get dataloaders
    tr_dl = DataLoader(dataset=tr_ds, batch_size=bs, shuffle=True, num_workers=ns)
    val_dl = DataLoader(dataset=val_ds, batch_size=bs, shuffle=False, num_workers=ns)
    test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=ns)

    return ds, tr_dl, val_dl, test_dl, n_cls
