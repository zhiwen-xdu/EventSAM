def bb_iou(boxA, boxB):
    # boxA/boxB: [x_left_up,y_left_up,w,h]
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def remove_small_mask(masks):
    opposite_masks = sorted(masks, key=lambda x: x['area'], reverse=False)  # 面积从小到大
    remove_mask_idx = []
    save_masks = []
    for i in range(len(opposite_masks)):
        bbox_area = opposite_masks[i]['area']
        if bbox_area < 144:
            remove_mask_idx.append(i)
    for i in range(len(opposite_masks)):
        if i not in remove_mask_idx:
            save_masks.append(opposite_masks[i])
    return save_masks


def remove_repeat_mask(masks):
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True) # 面积从大到小
    opposite_masks = sorted(masks, key=lambda x: x['area'], reverse=False)  # 面积从小到大
    interval_masks = len(masks)-1
    remove_mask_idx = []
    save_masks = []
    for i in range(len(sorted_masks)):
        for j in range(len(opposite_masks)):
            if i+j < interval_masks:
                large_bbox = sorted_masks[i]["bbox"]   # [x_left_up,y_left_up,w,h]
                small_bbox = opposite_masks[j]["bbox"]
                large_bbox_left,large_bbox_right = large_bbox[0],large_bbox[0] + large_bbox[2]
                large_bbox_up,large_bbox_down = large_bbox[1],large_bbox[1] + large_bbox[3]
                small_bbox_x,small_bbox_y = small_bbox[0]+small_bbox[2]/2,small_bbox[1]+small_bbox[3]/2
                if (large_bbox_left < small_bbox_x < large_bbox_right) and (large_bbox_up < small_bbox_y < large_bbox_down):
                    iou = bb_iou(small_bbox,large_bbox)
                    if iou > 0.1:
                        remove_mask_idx.append(j)

    remove_mask_idx = set(remove_mask_idx)
    remove_mask_idx = list(remove_mask_idx)
    for i in range(len(opposite_masks)):
        if i not in remove_mask_idx:
            save_masks.append(opposite_masks[i])
    return save_masks


def obtain_mask_image(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        # [3,]
        color_mask = np.random.random(3)
        img[m] = color_mask
    return img
