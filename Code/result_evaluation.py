import numpy as np
from collections import defaultdict
from ultralytics.engine.results import Results 
# from ultralytics import YOLO
from pathlib import Path
import pickle


def calculate_distance(pt1:tuple[float], pt2:tuple[float]) -> float:
    """ Calculates the distance between 2 points """
    dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    return dist 


def get_ground_from_label_file(label_folder_path:str|Path, image_name:str, class_list:list[str]) -> list[dict]:
    """ Gets the ground entries data from label's file
    """
    assert Path(label_folder_path).is_dir(), "Cannot find folder"
    assert (Path()/label_folder_path/image_name).is_file(), "File not found"
    with open(Path()/label_folder_path/image_name) as f:
        label = f.read().splitlines()
        label_list = [[int(val) if i == 0 else float(val) for i, val in enumerate(entry.split(" "))] for entry in label]
        ground_truth = [{'class':class_list[cell[0]], 'x':cell[1], 'y': cell[2]} for cell in label_list]
        return ground_truth


def calculate_class_metrics(pred_results_list:list[Results], label_folder_path:Path) -> dict:
    """ Calculates the total number of true-positive, true-native & false-positives 
        from a results list by comparing it to the ground truth for classes only 
        ================================================================================
        For every result box value -> finds the closest ground value using Euclidean distance, then:
        - If there are more predictions than ground values -> false-positive
        - If prediction != ground -> false-positive
        - If prediction == ground -> true-positive
        - If more ground values than predictions -> false-negative
    """
    class_metrics = {cls_name:{'true_positive':0, 'false_positive':0, 'false_negative':0} for cls_name in pred_results_list[0].names.values()}
    
    for pred_result in pred_results_list:
        image_name = Path(pred_result.path).parts[-1].split(".")[0] + ".txt"
        ground_truth = get_ground_from_label_file(label_folder_path=label_folder_path,
                                                  image_name=image_name,
                                                  class_list=list(pred_result.names.values()))

        preds = [{'class':pred_result.names[cls],'x':box[0],'y': box[1]} 
                for cls, box in zip(pred_result.boxes.cls.numpy().astype(np.uint8), 
                                    pred_result.boxes.xywhn.numpy())]
        # calculate the distace between all predictions and ground
        closest_ground_pred_pairing = defaultdict(int)
        for i, pred in enumerate(preds):
            closest_ground_idx = None
            closest_ground_dist = np.inf
            for j, ground in enumerate(ground_truth):
                dist = calculate_distance(pt1=(pred['x'], pred['y']), pt2=(ground['x'], ground['y']))
                if dist < closest_ground_dist:
                    closest_ground_dist = dist
                    closest_ground_idx = j
            closest_ground_pred_pairing[closest_ground_idx] = i
        # Check metrics
        preds_with_ground = closest_ground_pred_pairing.values()
        preds_without_ground = list(set(np.arange(len(preds))) - set(preds_with_ground))
        # More predictions than ground -> false positives
        for lonely_pred in preds_without_ground:
            lonely_pred_class = preds[lonely_pred]['class']
            class_metrics[lonely_pred_class]['false_positive'] += 1
        for ground_idx, pred_idx in closest_ground_pred_pairing.items():
            ground_class = ground_truth[ground_idx]['class']
            pred_class = preds[pred_idx]['class']
            # if pred class == ground class -> true positive
            if ground_class == pred_class:
                class_metrics[pred_class]['true_positive'] += 1
            # if pred class != ground class -> false positive
            else:
                class_metrics[pred_class]['false_positive'] += 1
        # More ground than pred -> false negatives
        lonely_grounds = list(set(np.arange(len(ground_truth))) - set(closest_ground_pred_pairing.keys()))
        for ground_idx in lonely_grounds:
            ground_class = ground_truth[ground_idx]['class']
            class_metrics[ground_class]['false_negative'] += 1

    return class_metrics


def get_precision_recall_f1(class_metrics:dict) -> tuple[float, float, float]:
    """ Returns the precision, recall and F1 score for a class metrics dict 
    """
    true_positives = [class_metrics[class_name]['true_positive'] for class_name in class_metrics]
    false_positives = [class_metrics[class_name]['false_positive'] for class_name in class_metrics]
    false_negatives = [class_metrics[class_name]['false_negative'] for class_name in class_metrics]
    try:
        overall_precision = round(number=sum(true_positives) / (sum(true_positives) + sum(false_positives)), ndigits= 4)
    except ZeroDivisionError:
        overall_precision = 0
    try:
        overall_recall = round(number = sum(true_positives) / (sum(true_positives) + sum(false_negatives)), ndigits= 4)
    except ZeroDivisionError:
        overall_recall = 0
    try:
        overall_f1_score = round(number=2 * (overall_precision * overall_recall) / (overall_precision + overall_recall), ndigits= 4)
    except ZeroDivisionError:
        overall_f1_score = 0
    return overall_precision, overall_recall, overall_f1_score   


if __name__ == "__main__":
    # model = YOLO(model='../Yolo/best.pt')

    yolo_folder_path = Path().cwd().parent / "Yolo"
    test_images_folder_path = yolo_folder_path/'test'/'images'
    image_names = ['A014_18Z_T15092_MID_x40_z0_i01j42.jpg', 'A014_18Z_T15092_MID_x40_z0_i02j06.jpg','A014_18Z_T15092_MID_x40_z0_i09j08.jpg']
    test_images = [test_images_folder_path/image_name for image_name in image_names]

    # results = model.predict(source = test_images)

    # with open(file='results_list.pickle', mode='wb') as f:
    #     pickle.dump(obj=results, file=f)

    with open(file=Path().cwd() /'results_list.pickle', mode='rb') as f:
        results = pickle.load(f)

    cm = calculate_class_metrics(pred_results_list=results, label_folder_path=yolo_folder_path/'test'/'labels')

    print(cm)