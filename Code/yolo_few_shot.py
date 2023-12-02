import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import pickle
from dataclasses import dataclass

import functions as funcs

from ultralytics import YOLO
from ultralytics.engine.results import Results 

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input




def create_few_shot_folders(folder_path:Path, image_df:pd.DataFrame, dir_name:str='Few_Shot') -> Path:
    """ Creates a folder structure for few shot learning
        Returns the path to the created folder
    """
    assert not (Path() / folder_path / dir_name).is_dir(), \
        "Folder already exists"
    
    few_shot_dir_path = Path()/ folder_path/dir_name
    class_initials = image_df['wbc_class'].unique()
    for class_initial in class_initials:
        class_dir = few_shot_dir_path/class_initial
        class_dir.mkdir(parents=True, exist_ok=True)

    few_shot_dir_path = folder_path/dir_name
    return few_shot_dir_path



def create_dataset(shot:int, folder_path:Path, image_df:pd.DataFrame) -> None:
    """ Creates an images dataset based on classes (way) and number of support set images (shot) 
        Extracts the ROI of cell from the images
        5-Way,2-Shot -> 5 classes and 2 images in the support set
    """
    np.random.seed(79)
    # Check to see if folders are empty
    assert not all([any(Path(folder_path/cls_int).iterdir()) 
                    for cls_int in image_df['wbc_class'].unique()]), "Folders are not empty"

    for class_initials in image_df['wbc_class'].unique():
        
        indices:np.ndarray = image_df[image_df['wbc_class'] == class_initials].index.values
        assert indices.size >= shot, "shot > indices" 
        chosen_indices = np.random.choice(a=indices, size=shot, replace=False)
        
        for idx in chosen_indices:
            img_name = image_df.loc[idx, 'Image File']
            image_path = funcs.get_image_path(image_file_name=img_name, total_df=image_df)
            img = funcs.get_image_from_path(file_path=image_path)
            x_min, x_max, y_min, y_max = funcs.get_bbox_from_xcoord_ycoord(x_coord=image_df.loc[idx].x_coord,
                                                                           y_coord=image_df.loc[idx].y_coord)
            roi = img[y_min:y_max, x_min:x_max]
            class_dir:Path = folder_path/class_initials
            assert class_dir.is_dir(), "Folder not found"
            funcs.save_image_to_path(image=roi, file_path=class_dir/img_name)


def preprocess_image( img:np.ndarray) -> np.ndarray:
    """ Pre-processes the image before feature extraction """
    img = cv2.resize(src=img, dsize=(224,224))
    img = preprocess_input(img)
    return img


@dataclass
class PredictionResult:
    predicted_classes:list
    predicted_scores:list
    boxes_xyxy:list
    center_x_y_norm:list
    path:Path


class Yolo_N_Shot:
    def __init__(self, yolo_weights_path:Path, 
                 class_list:list[str],
                 feature_extractor:Model,
                 prototypes:np.ndarray) -> None:
        self.yolo_weights_path = yolo_weights_path
        self.yolo_model = YOLO(model=self.yolo_weights_path)
        self.class_list = class_list
        self.feature_extractor = feature_extractor
        self.prototypes = prototypes
    
    
    def predict(self, image_path:Path) -> PredictionResult | None:
        """ Takes an image and returns a prediction result """
        # image = funcs.get_image_from_path(file_path=image_path)
        assert isinstance(image_path, Path), "image_path must be a Path object"
        assert image_path.is_file() and image_path.suffix in ['.jpeg', '.jpg', '.png'], "File not found"
        image = cv2.imread(filename=image_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yolo_pred:Results = self.yolo_model.predict(source=image_path)
        if yolo_pred == None:
            return None 
        center_x_y = [[vals[0], vals[1]] for vals in [box for box in yolo_pred[0].boxes.xywhn.numpy()]]
        boxes = [box for box in yolo_pred[0].boxes.xyxy.numpy().astype(np.int32)]
        # x_min, y_min, x_max, y_max = box
        rois = [image[box[1]:box[3], box[0]:box[2]] for box in boxes]
        rois_preprocessed = [preprocess_image(img=roi) for roi in rois]
        rois_exapanded = [np.expand_dims(roi, axis=0) for roi in rois_preprocessed]
        roi_features = [self.feature_extractor.predict(roi) for roi in rois_exapanded]            
        distances = [np.sqrt(np.sum((self.prototypes - roi_feature) ** 2, axis=-1)) for roi_feature in roi_features]        
        softmax_probs = [np.exp(-distance) / np.sum(np.exp(-distance)) for distance in distances]
        prediction_index = [np.argmax(softmax_prob) for softmax_prob in softmax_probs]
        predicted_classes = [self.class_list[idx] for idx in prediction_index]
        predicted_scores = [round(softmax_prob[np.argmax(softmax_prob)],3) for softmax_prob in softmax_probs]
        return PredictionResult(boxes_xyxy=boxes, 
                                predicted_classes=predicted_classes, 
                                predicted_scores=predicted_scores,
                                center_x_y_norm = center_x_y,
                                path=image_path)


def get_prototypes(few_shot_folder_path:Path, feature_extractor:Model, class_list:list[str]) -> np.ndarray:
    """ Returns the class prototypes for few shot inference
    """
    if (few_shot_folder_path/'prototypes_2_shot.pickle').exists():
        with open(file=few_shot_folder_path/'prototypes_2_shot.pickle', mode= 'rb') as f:
            prototypes = pickle.load(f)
            return prototypes 
    else:
        prototypes = np.empty(shape=(len(class_list), 4096))
        # The few shot folders are labelled using class initials & stored in alphabetical order
        class_initial = [class_name[0] for class_name in class_list]
        for i, class_initial in enumerate(class_initial):
            class_dir = few_shot_folder_path/class_initial
            assert class_dir.is_dir(), "Folder not found"
            class_images = [funcs.get_image_from_path(file_path=image_path) 
                        for image_path in class_dir.iterdir() 
                        if (image_path.is_file() & (image_path.suffix in ['.jpg', '.jpeg']))]
            class_images = np.array([preprocess_image(img) for img in class_images])
            class_features = feature_extractor.predict(class_images)
            class_prototype = np.mean(class_features, axis=0)
            prototypes[i] = class_prototype

        with open(file=few_shot_folder_path/'prototypes_2_shot.pickle', mode='wb') as f:
            pickle.dump(obj=prototypes, file=f)
        
        return prototypes



if __name__ == "__main__":

    total_df = funcs.get_image_df()
    total_df = funcs.train_test_split_df(total_df=total_df)

    train_df = pd.DataFrame()
    yolo_folder_path = Path().cwd().parent/'Yolo' 
    yolo_images_training_folder = yolo_folder_path/'train'/'images'

    train_file_names = [file.parts[-1] for file in yolo_images_training_folder.iterdir() if (file.is_file()) & (file.suffix == '.jpg')]
    train_df['Image File'] = train_file_names
    train_df = pd.merge(left=train_df, right=total_df[['Image File', 'x_coord', 'y_coord','wbc_class','set']], on='Image File', how='left')

    ROOT_PATH = Path().cwd().parent
    
    try:
        few_shot_dir_path = create_few_shot_folders(folder_path=ROOT_PATH, image_df=train_df)
    except AssertionError as e:
        few_shot_dir_path = Path()/ ROOT_PATH/"Few_Shot"

    try:
        create_dataset(shot=2, folder_path=few_shot_dir_path, image_df=train_df)
    except AssertionError as e:
        print(e, " -> Skipping dataset creation")
    
    wbc_class = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]

    model = VGG16(weights='imagenet', include_top=True)
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    few_shot_dir_path = Path()/ ROOT_PATH/"Few_Shot"
    prototypes = get_prototypes(few_shot_folder_path=few_shot_dir_path, feature_extractor=feat_extractor, class_list=wbc_class)

    yolo_folder_path = ROOT_PATH / "Yolo"
    val_images_folder_path = yolo_folder_path/'val'/'images'
    val_img_path = val_images_folder_path /'A014_18Z_T15092_MID_x40_z0_i32j36.jpg'

    y2s = Yolo_N_Shot(yolo_weights_path='../Yolo/best.pt', 
                      class_list=wbc_class, 
                      feature_extractor=feat_extractor, 
                      prototypes=prototypes)
    
    res = y2s.predict(image_path=val_img_path)
    print(f"{res=}")