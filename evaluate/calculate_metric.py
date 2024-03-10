import cv2
import numpy as np
from glob import glob
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time
import os
from multiprocessing import Pool


def get_unique_colors(img):
    return set(tuple(v) for m2d in img for v in m2d)

def mask_from_color(img, color):
    return np.all(img == np.array(color).reshape(1, 1, 3), axis=2).astype(np.uint8)

def unique_colors_and_counts_in_part(img_part):
    # Flatten the part and get unique colors and their counts
    colors, inverse_indices = np.unique(img_part.reshape(-1, 3), axis=0, return_inverse=True)
    counts = np.bincount(inverse_indices)
    return colors, counts

def combine_and_filter_colors(results, min_size):
    # Initialize a dictionary to keep the counts of each unique color
    color_counts = {}

    # Combine the results from each part
    for colors, counts in results:
        for color, count in zip(colors, counts):
            color_tuple = tuple(color)
            if color_tuple in color_counts:
                color_counts[color_tuple] += count
            else:
                color_counts[color_tuple] = count

    # Filter the colors by the min_size and return them
    filtered_colors = [color for color, count in color_counts.items() if count >= min_size]
    return np.array(filtered_colors)

def get_unique_colors(img, min_size=12, num_processes=None):
    if num_processes is None:
        num_processes = os.cpu_count()

    # Calculate the size of each split, ensuring every row is included
    total_rows = img.shape[0]
    split_height = total_rows // num_processes
    extra_rows = total_rows % num_processes

    # Divide the image into parts, accounting for any extra rows
    img_parts = [img[i * split_height + min(i, extra_rows):(i + 1) * split_height + min(i + 1, extra_rows)] for i in range(num_processes)]

    # Process each part in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(unique_colors_and_counts_in_part, img_parts)

    # Combine colors and counts, and filter globally based on min_size
    unique_colors_filtered = combine_and_filter_colors(results, min_size)
    return unique_colors_filtered

def mask_from_color(img, color):
    return np.all(img == np.array(color).reshape(1, 1, 3), axis=2).astype(np.uint8)

def calculate_metric(mask1, mask2):
    # mask1:TP+FN, mask2: TP+FP
    intersection = np.logical_and(mask1, mask2)   # TP
    union = np.logical_or(mask1, mask2)           # TP+FN+FP
    precision = np.sum(intersection) / np.sum(mask2)
    recall = np.sum(intersection) / np.sum(mask1)
    iou = np.sum(intersection) / np.sum(union)
    return precision,recall,iou


def best_matching_segment(args):
    img1_color, img2, unique_colors_img2 = args
    max_iou = 0
    max_precision = 0
    max_recall = 0
    mask_img1 = mask_from_color(img1, img1_color)
    mask1_size = np.sum(mask_img1)
    for color2 in unique_colors_img2:
        mask_img2 = mask_from_color(img2, color2)
        precision,recall,iou = calculate_metric(mask_img1, mask_img2)
        if iou > max_iou:
            max_iou = iou
            max_precision = precision
            max_recall = recall

    return max_precision, max_recall, max_iou, mask1_size


def calculate_one_sample(img1,img2):
    unique_colors_img1 = get_unique_colors(img1)
    unique_colors_img2 = get_unique_colors(img2)
    # Use ProcessPoolExecutor to parallelize the computation
    tasks = [(color1, img2, unique_colors_img2) for color1 in unique_colors_img1]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(best_matching_segment, tasks))

    # Extract IoUs and sizes
    precisions, recalls, ious, sizes = zip(*results)

    # Calculate the weighted IoU
    total_size = np.sum(sizes)
    average_precision = np.mean(precisions)
    average_recall = np.mean(recalls)
    average_iou = np.mean(ious)
    weighted_iou = np.sum(np.array(ious) * np.array(sizes) / total_size)

    return average_precision,average_recall,average_iou,weighted_iou

# Note: the gt_dir denotes the groundtruth mask dir for testing (e.g. ./data/Groundtruths/MVSEC-SEG/).
# Note: the val_mask_txt denotes the metric saver (e.g. ./data/Predictions/MVSEC-SEG.txt).
gt_dir = "..."
val_mask_txt = "..."
f1 = open(val_mask_txt, 'a')

all_precisions = []
all_recalls = []
all_ious = []
all_weight_ious = []
# Note: seq e.g. indoor_flying1
for seq in sorted(glob(gt_dir+"*"))[:]:
    seq_name = seq.split("/")[-1]

    seq_ious = []
    seq_precisions = []
    seq_recalls = []
    seq_weight_ious = []
    # Note: e.g. gt_mask_path=./data/Groundtruths/MVSEC-SEG/indoor_flying1/0488.png; val_mask_path=./data/Predictions/MVSEC-SEG/indoor_flying1/0488.png;
    for gt_mask_path in sorted(glob(seq+"/*.png")):
        val_mask_path = gt_mask_path.replace("Groundtruths","Predictions")
        img1 = cv2.imread(gt_mask_path)
        img2 = cv2.imread(val_mask_path)

        average_precision,average_recall,average_iou,weighted_iou = calculate_one_sample(img1, img2)
        # print(average_precision,average_recall,average_iou,weighted_iou)

        seq_precisions.append(average_precision)
        seq_recalls.append(average_recall)
        seq_ious.append(average_iou)
        seq_weight_ious.append(weighted_iou)

    seq_precision = np.mean(seq_precisions)
    seq_recall = np.mean(seq_recalls)
    seq_iou = np.mean(seq_ious)
    seq_weight_iou = np.mean(seq_weight_ious)
    print(seq_name+":",f"AP: {seq_precision:.3f}", f"AR: {seq_recall:.3f}", f"mIoU: {seq_iou:.3f}", f"aIoU: {seq_weight_iou:.3f}")
    all_precisions.append(seq_precision)
    all_recalls.append(seq_recall)
    all_ious.append(seq_iou)
    all_weight_ious.append(seq_weight_iou)
    f1.write(seq_name + ":" + str(seq_precision) + "," + str(seq_recall) + "," + str(seq_iou) + "," + str(seq_weight_iou) + "\n")


mean_precision = np.mean(all_precisions)
mean_recall = np.mean(all_recalls)
mean_iou = np.mean(all_ious)
mean_weight_iou = np.mean(all_weight_ious)
print(f"AP: {mean_precision:.3f}", f"AR: {mean_recall:.3f}", f"mIoU: {mean_iou:.3f}",f"aIoU: {mean_weight_iou:.3f}")
f1.write("all:" + str(mean_precision) + "," + str(mean_recall) + "," + str(mean_iou) + "," + str(mean_weight_iou) + "\n")
f1.close()
