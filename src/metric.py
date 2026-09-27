import torch

def calculate_iou(boxA, boxB, epsilon=1e-5):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / (area_combined+epsilon)
    return iou

def calculate_ap(precision, recall):
    # Calculate Average Precision (AP) from precision and recall values
    mrec = torch.cat((torch.zeros((1,)), recall, torch.ones((1,))))
    mpre = torch.cat((torch.zeros((1,)), precision, torch.zeros((1,))))

    for i in range(mpre.size(0) - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

    i = torch.nonzero(mrec[1:] != mrec[:-1]) + 1
    ap = torch.sum((mrec[i] - mrec[i - 1]) * mpre[i])

    return ap

def calculate_precision_recall(predictions, targets, iou_threshold=0.5):
    # Calculate precision and recall for a set of predictions and targets
    true_positives = torch.zeros(len(predictions))
    false_positives = torch.zeros(len(predictions))
    false_negatives = torch.zeros(len(targets))

    for i, pred in enumerate(predictions):
        iou_max = 0
        for j, target in enumerate(targets):
            iou = calculate_iou(pred, target)
            if iou > iou_max:
                iou_max = iou
                j_max = j

        if iou_max >= iou_threshold:
            true_positives[i] = 1
            false_negatives[j_max] = 0
        else:
            false_positives[i] = 1

    cumulative_true_positives = torch.cumsum(true_positives, dim=0)
    cumulative_false_positives = torch.cumsum(false_positives, dim=0)
    cumulative_false_negatives = torch.cumsum(false_negatives, dim=0)

    precision = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives + 1e-16)
    recall = cumulative_true_positives / (cumulative_true_positives + cumulative_false_negatives + 1e-16)

    return precision, recall

def calculate_map(predictions, targets, iou_threshold=0.5):
    # Calculate Mean Average Precision (mAP) for a set of predictions and targets
    ap_values = []

    for c in range(len(predictions[0])):
        precision, recall = calculate_precision_recall(predictions[:, c, :], targets[:, c, :], iou_threshold)
        ap = calculate_ap(precision, recall)
        ap_values.append(ap.item())

    return sum(ap_values) / len(ap_values)
