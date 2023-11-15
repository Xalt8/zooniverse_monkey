from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import shutil
from image_annotation import Cell, ImageAnnotation


def validate_file_path(file_path:str|Path) -> str:
    """ Checks to see if the file path is valid 
    """
    if not isinstance(file_path, (str, Path)):
        raise ValueError(f"{file_path} must be a str or Path object")
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot find this file -> {file_path} ")
    return file_path.as_posix()


def get_image_from_path(file_path:str | Path) -> np.ndarray:
    """ Takes an file path and returns an image as array 
    """
    file_path = validate_file_path(file_path= file_path)
    img = cv2.imread(filename=file_path) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_bad_shapes() -> set[str]:
    """ Returns a set with the file names of images that are not in the shape (1000,1000,3) 
    """
    if (Path() / r'bad_shapes.pickle').is_file():
        with open(file='bad_shapes.pickle', mode= 'rb') as f:
            bad_shapes = pickle.load(f)
            return bad_shapes 

    bad_shape = []

    image_folders = [r"../AI Training Sets-20231024T092556Z-001\AI Training Sets\Set4-1_WBC_Images", 
                     r"../AI Training Sets-20231024T092556Z-001\AI Training Sets\Set5-1_WBC_Images"]

    for image_folder in image_folders:
        for file_path in (Path() /image_folder).iterdir():
            img = get_image_from_path(file_path=file_path.as_posix())
            if img.shape == (1000,1000,3):
                continue
            else:
                bad_shape.append(file_path.parts[-1])        
    
    with open(file='bad_shapes.pickle', mode='wb') as f:
        pickle.dump(obj=set(bad_shape), file=f)

    return set(bad_shape)


def get_image_df() -> pd.DataFrame:
    """Returns a dataframe with all the image data """

    set4_df = pd.read_excel(r"../AI Training Sets-20231024T092556Z-001\AI Training Sets\Set4-1_TrainingData_20210526.xlsx", index_col=None)
    set4_df = set4_df[set4_df.columns[:-2]]
    set4_df["Animal Date"] = pd.to_datetime(set4_df["Animal Date"])

    set5_df = pd.read_excel(r"../AI Training Sets-20231024T092556Z-001\AI Training Sets\Set5-1_TrainingData_20220413.xlsx", index_col=None)
    set5_df = set5_df[set5_df.columns[:-2]]
    set5_df["Animal Date"] = pd.to_datetime(set5_df["Animal Date"])

    df1 = set4_df.iloc[:,:-1]
    df1['set'] = "Set4"
    df2 = set5_df.iloc[:,:-1]
    df2['set'] = "Set5"
    total_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    total_df = total_df.rename(columns={"X - Coordinate (pixels)": "x_coord", "Y - Coordinate (pixels)": "y_coord", "WBC Classification": "wbc_class"})
    total_df['x_coord'] = total_df['x_coord'].astype(int)
    total_df['y_coord'] = total_df['y_coord'].astype(int) 
    total_df['wbc_class'] = total_df['wbc_class'].astype('category')
    total_df['set'] = total_df['set'].astype('category')
    bad_shapes = get_bad_shapes()
    total_df = total_df[~total_df['Image File'].isin(bad_shapes)]
    total_df.reset_index(drop=True, inplace=True)
    return total_df


def get_image_path(image_file_name:str, total_df:pd.DataFrame) -> str:
    """ Takes an image name and returns the path to the image in the correct folder
    """
    image_folder_dict = {'Set5': Path()/ r"../AI Training Sets-20231024T092556Z-001\AI Training Sets\Set5-1_WBC_Images",
                         'Set4': Path() / r"../AI Training Sets-20231024T092556Z-001\AI Training Sets\Set4-1_WBC_Images"}
    
    set_val = total_df[total_df['Image File'] == image_file_name]['set'].unique()[0]
    return (image_folder_dict[set_val] / image_file_name).as_posix()


def plot_image_file(ax:plt.axes, image_file_name:str, total_df:pd.DataFrame) -> plt.axes:
    """ Takes an image file name, looks up the dataframe for all instances 
        of WBCs in the image file and highlights the cells in the image 
        
        Usage:
        fig, ax = plt.subplots()
        plot_image_file(ax=ax,image_file_name='B014_77I_T19095_RT_x40_z0_i02j07.jpg', total_df=total_df)
        plt.show()    
    """
    image_file_df = total_df[total_df['Image File'] == image_file_name]
    image_path = get_image_path(image_file_name=image_file_name,
                                total_df=total_df)
    img = get_image_from_path(image_path)
    colors = {'N' : "red", 'L' : "yellow", 'M' : "lightgreen", 'E' : "lightblue", 'B' : "magenta"}
    ax.imshow(img)
    ax.set_title(image_file_name, fontsize=8)
    ax.set_xticks([]), ax.set_yticks([])
    for row_idx in image_file_df.index:
        x,y,cell = image_file_df.loc[row_idx, ['x_coord','y_coord','wbc_class']]        
        ax.scatter(x=x,y=y,s=500, label=cell, facecolors='none', edgecolors=colors.get(cell, 'black'))
    ax.legend(bbox_to_anchor=(0.5,-0.10), loc='lower center', ncol=5)
    return ax 


def get_normalized_coordinates(x:int, y:int, 
                               image_width:int = 1000, 
                               image_height:int= 1000, 
                               bbx_size:tuple[int, int] = (70,70)) -> tuple[float, float, float, float]:
    """ Returns the normalized center_x, center_y, width & height given the center x,y
    """
    center_x_normalized = x / image_width
    center_y_normalized = y / image_height
    width_normalized = bbx_size[0] / image_width
    height_normalized = bbx_size[1] / image_height
    return center_x_normalized, center_y_normalized, width_normalized, height_normalized   


def get_bbox_coords_from_normalized_coords(center_x_normalized:float,
                                           center_y_normalized:float,
                                           width_normalized:float,
                                           height_normalized:float,
                                           image_width:int = 1000,
                                           image_height:int = 1000) -> tuple[int, int, int, int]:
    """ Takes a tuple of normalized coordinates and returns the bounding box coordinates -> used for plotting
    """
    # Reverse normalization to get bounding box coordinates in image space
    center_x = int(center_x_normalized * image_width)
    center_y = int(center_y_normalized * image_height)
    width = int(width_normalized * image_width)
    height = int(height_normalized * image_height)
    
    # Calculate bounding box coordinates
    x_min = int(center_x - (width / 2))
    y_min = int(center_y - (height / 2))
    x_max = int(center_x + (width / 2))
    y_max = int(center_y + (height / 2))
    return x_min, x_max, y_min, y_max 


def get_image_annotation(image_name:str, total_df:pd.DataFrame) -> ImageAnnotation:
    """ Takes an image name a returns a ImageAnnotation object
    """
    image_annotation = ImageAnnotation(image_name = image_name, 
                                       image_path = get_image_path(image_file_name=image_name, 
                                                                  total_df=total_df))
    image_df_vals = total_df[total_df["Image File"] == image_name][['x_coord', 'y_coord', 'wbc_class']].values
    for x, y, wbc_class in image_df_vals:
        center_x_normalized, center_y_normalized, width_normalized, height_normalized = get_normalized_coordinates(x=x, y=y)
        cell = Cell(wbc_class = wbc_class, 
                    x_coord = x, 
                    y_coord = y, 
                    center_x_normalized = center_x_normalized,
                    center_y_normalized = center_y_normalized,
                    width_normalized = width_normalized,
                    height_normalized = height_normalized)
        image_annotation.cells = [cell] if image_annotation.cells is None else image_annotation.cells + [cell]
    return image_annotation


def train_test_split_indicies(indices:np.ndarray, 
                              train_percent:float=0.70, 
                              val_percent:float = 0.20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Shuffles and splits an array of index values into train and val 
        and remaining into test indices
    """
    assert len(indices) > 5, "Length of indices must > 5" 
    train_idx_num = int(len(indices) * train_percent) 
    val_idx_num = int(len(indices) * val_percent)
    np.random.shuffle(indices)
    return indices[:train_idx_num], indices[train_idx_num:train_idx_num + val_idx_num], indices[train_idx_num + val_idx_num:]


def set_train_test_val(row_index:int, cat:str, total_df:pd.DataFrame) -> None:
    """ Assigns a row value to a given train, test or val category 
        Checks for other instances of the same image file and assigns it to 
        to the same category
    """
    assert cat in ['Train', 'Test', 'Val'], 'cat should be either Train, Test, or Val' 
    
    image_file = total_df.loc[row_index, 'Image File']
    instances = total_df[total_df['Image File'] == image_file]
    for row_idx in instances.index.values:
        if total_df.loc[row_idx, 'train_test_val'] is np.nan:
            total_df.loc[row_idx, 'train_test_val'] = cat

  

def train_test_split_df(total_df:pd.DataFrame) -> pd.DataFrame:
    """ Adds a train_test_val column to the dataframe. 
        For every wbc cell class -> gets the index values in the dataframe,
        divides them into train, val & test and assigns each entry to a category
    """
    total_df['train_test_val'] = pd.Series(dtype='category')
    categories = ['Train', 'Val', 'Test']
    total_df['train_test_val'] = pd.Categorical(total_df['train_test_val'], categories=categories)

    for wbc_class in np.sort(total_df['wbc_class'].unique()):
        class_df:pd.DataFrame = total_df[total_df['wbc_class'] == wbc_class]
        class_indices = class_df.index.values
        if wbc_class == 'B':
            set_train_test_val(row_index=class_indices[0], cat=categories[0], total_df=total_df)
            set_train_test_val(row_index=class_indices[1], cat=categories[0], total_df=total_df)
            set_train_test_val(row_index=class_indices[2], cat=categories[1], total_df=total_df)
            set_train_test_val(row_index=class_indices[3], cat=categories[2], total_df=total_df) 
        else:
            split_indicies = train_test_split_indicies(class_indices)
            for indicies, cat in zip(split_indicies, categories):
                for idx in indicies: 
                    set_train_test_val(row_index=idx, cat=cat, total_df=total_df)
    return total_df


def make_yolo_directory(path:str = None) -> None:
    """ Creates a directory structure for Yolo training
    """
    if path is None:
        path = Path().cwd()

    if (Path() / path / "Yolo").is_dir():
        return
    
    # Define the root folder and its subfolders
    root_folder = Path(path/"Yolo")
    subfolders = ['train', 'val', 'test']

    # Create the directory structure
    for folder in subfolders:
        sub_folder = root_folder / folder
        images_folder = sub_folder / 'images'
        labels_folder = sub_folder / 'labels'
        
        sub_folder.mkdir(parents=True, exist_ok=True)
        images_folder.mkdir(parents=True, exist_ok=True)
        labels_folder.mkdir(parents=True, exist_ok=True)


def make_annotation_file(image_annotation:ImageAnnotation, file_path:str) -> None:
    """ Takes an ImageAnnotation object and creates a txt file with the image annotations
    """
    wbc_list = ['N', 'L', 'M', 'E', 'B']
    with open(file=file_path, mode='w') as f:
        for cell in image_annotation.cells:
            line = (f"{wbc_list.index(cell.wbc_class)} "
                    f"{cell.center_x_normalized} "
                    f"{cell.center_y_normalized} "
                    f"{cell.width_normalized} "
                    f"{cell.height_normalized}\n")
            f.write(line)


def send_images_annotations(total_df:pd.DataFrame, yolo_folder_path:str) -> None:
    """ For every image in the dataframe -> check the category and copies the image file 
        from the source to corresponding the Yolo images folder and creates an annotations txt
        file in the labels folder
    """
    for file_name in total_df['Image File'].unique():
        image_file_df:pd.DataFrame = total_df[total_df['Image File'] == file_name]
        image_cat:str = image_file_df['train_test_val'].iloc[0]
        folder_map = {"Train":'train', "Val":'val', "Test": 'test'}
        cat_folder = Path()/yolo_folder_path/ folder_map[image_cat] 
        images_folder = cat_folder/'images'
        assert images_folder.is_dir(), "images_folder does not exist"
        source_image_path = get_image_path(image_file_name=file_name, total_df=total_df)
        assert Path(source_image_path).is_file(), "Incorrect image path"
        _ = shutil.copy(src=Path(source_image_path), dst=images_folder/file_name) 
        
        labels_folder = cat_folder/'labels'
        assert labels_folder.is_dir(), "labels_folder does not exist"
        image_annotation = get_image_annotation(image_name=file_name, total_df=total_df)
        label_file_name = file_name.split(".")[0] + ".txt"
        make_annotation_file(image_annotation=image_annotation, file_path=labels_folder/label_file_name)
    
    



if __name__ == "__main__":
    ...