from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from functions import get_image_from_path
from dump import save_image_to_path


def rotate_point_clockwise(x:int, y:int, image_size:tuple[int, int]) -> tuple[int, int]:
    ''' Takes a tuple of x,y points and returns the positions if
        rotated by 90 degrees clockwise
    '''
    assert len(image_size) == 2, "image_shape should be (height, width)"
    height, _ = image_size
    new_x = height - y
    new_y = x
    return new_x, new_y


Coordinates = tuple[int, int]


def rotate_image_point(img:np.ndarray, points:Coordinates) -> tuple[np.ndarray, Coordinates]:
    """ Takes an image and point coordinates -> rotates the image and returns the rotated 
        image and rotated point coordinates
    """
    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rotated_pts = rotate_point_clockwise(points[0], points[1], image_size=img.shape[:2])
    return rotated_img, rotated_pts


def image_augmented_check(image_info:pd.Series, aug_cvs_path:str) -> bool:
    """ Returns true if an image has been augmented """
    aug_df = pd.read_csv(aug_cvs_path)
    return image_info["Image File"] in set(aug_df['Original file'].values)


def write_to_aug_csv(row_dict:dict, aug_csv_path:str) -> None:
    """ Takes row values in the form of a dict and appends 
        it to the augmentions' csv """
    aug_csv_df:pd.DataFrame = pd.read_csv(filepath_or_buffer= aug_csv_path)
    aug_csv_df.loc[len(aug_csv_df)] = row_dict
    aug_csv_df.to_csv(aug_csv_path, index=False)


def rotate_augmentation(image_info:pd.Series, images_df:pd.DataFrame, aug_csv_path:str, aug_images_folder_path:str, bad_shape:list) -> None:
    """ Takes an image info row, rotates the image 3 times clockwise, adds the rotated
        images to a folder, add point coordinates to csv file. Checks to see if there are 
        more than 1 WBC in the same image and adds those to folder and file as well """

    if image_augmented_check(image_info= image_info, aug_cvs_path= aug_csv_path):
        print(f"{image_info['Image File']} has already been agumented")
        return

    if image_info['Image File'] in bad_shape: 
        return

    image_folder_dict = {'Set5': Path()/ r"AI Training Sets-20231024T092556Z-001\AI Training Sets\Set5-1_WBC_Images",
                         'Set4': Path() / r"AI Training Sets-20231024T092556Z-001\AI Training Sets\Set4-1_WBC_Images"}
    image_path = (image_folder_dict[image_info['set']] / image_info['Image File']).as_posix()
    img = get_image_from_path(file_path= image_path)
    assert img.shape == (1000,1000,3), "img not correct shape for augmentation"
    
    # Get all instances WBCs of the image
    filter_mask = images_df['Image File'] == image_info['Image File'] 
    multi_wbc_df:pd.DataFrame = images_df[["x_coord","y_coord","wbc_class"]][filter_mask].copy()
    
    for i in range(3): # Rotate the image 3 times clockwise
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        file_name = f"{image_info['Image ID']}_r{i}.jpg"
        file_path = (Path() / aug_images_folder_path / file_name).as_posix()
        save_image_to_path(image= img, file_path= file_path)
        
        for row_idx in multi_wbc_df.index:
            x,y = multi_wbc_df.loc[row_idx,["x_coord","y_coord"]]
            x,y = rotate_point_clockwise(x = x, y = y, image_size = img.shape[:2])
            multi_wbc_df.loc[row_idx,["x_coord","y_coord"]] = x,y
            row_dict = {
                'Original file': image_info['Image File'],
                'Image File': file_name,
                'x_coord': x,
                'y_coord': y,
                'wbc_class': multi_wbc_df.loc[row_idx,'wbc_class'],
                'set': image_info['set']
            }
            write_to_aug_csv(row_dict= row_dict, aug_csv_path= aug_csv_path)


if __name__ == "__main__":
    ...