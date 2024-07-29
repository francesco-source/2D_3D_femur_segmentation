import time
import numpy as np
import warnings
from tqdm import tqdm  # Import tqdm for progress bar
import torch
from PIL import Image
# Suppress the specific warning about DataLoader workers
warnings.filterwarnings("ignore", category=UserWarning, message=".*DataLoader.*max number of worker.*")
def calculate_volume_similarity(gtMask,predMask):
    gtMask = np.array(gtMask)
    predMask = np.array(predMask)
    tp = np.sum((gtMask == 1) & (predMask == 1))
    fp = np.sum((gtMask == 0) & (predMask == 1))
    fn = np.sum((gtMask == 1) & (predMask == 0))
    numerator = fn - fp
    numerator2 = abs(numerator)
    denominator = 2*tp + fp + fn
    if denominator == 0:
      return 0,0
    volume_similarity = 1 - (numerator/denominator)
    volume_similarityabs = 1 - (numerator2/denominator)
    return volume_similarity, volume_similarityabs

def calculateScores(gtMask, predMask):
    # Convert to NumPy arrays if they aren't already
    gtMask = np.array(gtMask)
    predMask = np.array(predMask)

    # Calculate the true positives (TP), false positives (FP), and false negatives (FN)
    tp = np.sum((gtMask == 1) & (predMask == 1))
    fp = np.sum((gtMask == 0) & (predMask == 1))
    fn = np.sum((gtMask == 1) & (predMask == 0))

    # Calculate IoU
    iou = tp / (tp + fp + fn)

    dice = np.sum(predMask[gtMask==1]) * 2.0 / (np.sum(predMask) + np.sum(gtMask))

    return iou, dice



def calculate_results(model,test_data):

  miou_test = []
  dicescores = []
  start_time = time.time()
  min_iou = 0
  current_images = []
  current_gt = []
  index = -1
  volume_similarities = []
  volume_similarity_abs = []
  list_mioU3D = []
  list_dice3D = []

  for idx, data in enumerate(tqdm(test_data)):
      im, gt, images_indexes = data

      if index != images_indexes:
        if index != -1:  # Ensure it's not the first iteration
            vol_sim , vol_sim_abs = calculate_volume_similarity(current_gt, current_images)
            volume_similarities.append(vol_sim)
            volume_similarity_abs.append(vol_sim_abs)
            miou3D, dice3D = calculateScores(current_gt,current_images)
            list_mioU3D.append(miou3D)
            list_dice3D.append(dice3D)
        index = images_indexes
        current_images = []
        current_gt = []



      # Get predicted mask
      with torch.no_grad():
          pred = torch.argmax(model(im.to(device)), dim=1)

      pred = np.array(pred.squeeze(0).cpu())  # Remove batch dimension
      gt = np.array(gt.squeeze(0).cpu()[0])

      Iou, dice = calculateScores(gt, pred)

      current_images.append(pred)
      current_gt.append(gt)

      miou_test.append(Iou)
      dicescores.append(dice)

  if current_gt and current_images:
          vol_sim , vol_sim_abs = calculate_volume_similarity(current_gt, current_images)
          volume_similarities.append(vol_sim)
          volume_similarity_abs.append(vol_sim_abs)
          miou3D, dice3D = calculateScores(current_gt,current_images)
          list_mioU3D.append(miou3D)
          list_dice3D.append(dice3D)
  end_time = time.time()  # End the timer

  execution_time = end_time - start_time
  return volume_similarities,volume_similarity_abs, miou_test, dicescores, execution_time, list_mioU3D,list_dice3D





def print_results(miou_test,dicescores,volume_similarities,vol_sim_abs,execution_time,list_mioU3D,list_dice3D):
  print("The MIOU for the segmentation over the test set resulted in:")
  print(np.round(np.array(miou_test).mean(),3), " +\- ",np.round(np.array(miou_test).std(),3) )
  print()
  print("The MIOU 3D for the segmentation over the test set resulted in:")
  print(np.round(np.array(list_mioU3D).mean(),3), " +\- ",np.round(np.array(list_mioU3D).std(),3) )
  print()
  print("The Dicescore for the segmentation over the test set resulted in:")
  print(np.round(np.array(dicescores).mean(),3), " +\- ", np.round(np.array(dicescores).std(),3))
  print()
  print("The Dicescore 3D for the segmentation over the test set resulted in:")
  print(np.round(np.array(list_dice3D).mean(),3), " +\- ", np.round(np.array(list_dice3D).std(),3))
  print()
  print("The volume similarity for the segmentation over the test set resulted in:")
  print(np.round(np.array(volume_similarities).mean(),3), " +\- ", np.round(np.array(volume_similarities).std(),3))
  print()

  print("The volume similarity abs for the segmentation over the test set resulted in:")
  print(np.round(np.array(vol_sim_abs).mean(),3), " +\- ", np.round(np.array(vol_sim_abs).std(),3))
  print()

  print()
  print("The time needed for the segmentation of 8 3D images is:")
  print(np.round(execution_time,2), "seconds.")


def calculate_group_averages(scores):
    # Convertiamo la lista in un array NumPy per facilitare la manipolazione
    scores_array = np.array(scores)

    # Calcoliamo la media dei gruppi di tre elementi
    num_groups = len(scores_array) // 3
    group_averages = []

    for i in range(num_groups):
        group = scores_array[i*3:(i+1)*3]  # Prendiamo il sottogruppo di tre elementi
        average = np.mean(group)  # Calcoliamo la media del gruppo
        group_averages.append(average)

    return group_averages