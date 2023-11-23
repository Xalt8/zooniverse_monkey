import cv2
import numpy as np
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from functions import get_image_path, get_image_from_path, ImageAnnotation, Cell, get_bbox_coords_from_normalized_coords


def get_binary_image(image_name: str, total_df:pd.DataFrame) -> np.ndarray:
    """ Takes an image file name applies filters and thresholding 
        and returns an inverted binary image
    """
    image_path = get_image_path(image_file_name = image_name, total_df = total_df)
    img = get_image_from_path(file_path= image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5)
    apply_clahe = clahe.apply(gray_img) +10
    _, binary_img = cv2.threshold(apply_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(binary_img, kernel, iterations=1)  
    inverted_binary = cv2.bitwise_not(dilated_img)
    return inverted_binary


def get_image_annotation2(image_name:str, total_df:pd.DataFrame) -> ImageAnnotation:
    """ Takes an image name a returns a ImageAnnotation object
    """
    image_annotation = ImageAnnotation(image_name = image_name, 
                                       image_path = get_image_path(image_file_name=image_name, 
                                                                  total_df=total_df))
    image_df_vals = total_df[total_df["Image File"] == image_name][['x_coord', 'y_coord', 'wbc_class']].values
    binary_image = get_binary_image(image_name= image_name, total_df= total_df)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        for image_vals in image_df_vals:
            x, y, wbc_class = image_vals
            if cv2.pointPolygonTest(contour=contour, pt=[x,y], measureDist=False) == 1:
                simplified_contour = cv2.approxPolyDP(curve=contour, epsilon=1, closed=True)
                cell = Cell(cell_class=wbc_class, cell_x_coord=x, cell_y_coord=y, cell_contour=simplified_contour)
                image_annotation.cells = [cell] if image_annotation.cells is None else image_annotation.cells + [cell]  
    return image_annotation


def save_image_to_path(image:np.ndarray, file_path:str|Path) -> None:
    """ Takes an image array converts it to BGR and saves to disk in RGB
    """
    if isinstance(file_path, Path):
        file_path = file_path.as_posix()
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename = file_path, img = img_bgr)


def parse_annotations(annotantions_json:json, image_folder_path:str) -> list[ImageAnnotation]:
    """ Takes an json file with polygon annotations and returns a list of ImageAnnotation objects
    """
    annotations = []
    for file_name in annotantions_json:
        cells = []
        for region in annotantions_json[file_name]['regions']:
            shape_attributes = annotantions_json[file_name]['regions'][str(region)]['shape_attributes']
            region_attribute_label = annotantions_json[file_name]['regions'][str(region)]['region_attributes']['label']
            cell_loc = np.array([[x,y] for x,y in zip(shape_attributes['all_points_x'], shape_attributes['all_points_y'])])
            cell = Cell(cell_class=region_attribute_label, cell_location=cell_loc)
            cells.append(cell)
        image_annotation = ImageAnnotation(image_name=file_name, 
                                           file_path=(image_folder_path/file_name).as_posix(),
                                           cells=cells)
        annotations.append(image_annotation)            
    return annotations


def display_annotations(image_name:str, total_df:pd.DataFrame, image_annotation:ImageAnnotation) -> None:
    """ Draws the image with the point coordinates and contours
    """
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    plot_image_file(ax=axs[0], image_file_name=image_name, total_df=total_df)
    image_path = get_image_path(image_file_name=image_name, total_df=total_df)
    img = get_image_from_path(image_path)
    for cell in image_annotation.cells:
        x_min, x_max, y_min, y_max = get_bbox_coords_from_normalized_coords(center_x_normalized = cell.center_x_normalized,
                                                                            center_y_normalized = cell.center_y_normalized,
                                                                            width_normalized = cell.width_normalized,
                                                                            height_normalized = cell.height_normalized)
        
        cv2.rectangle(img=img, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(0, 255, 0), thickness=2)
        axs[1].set_title('Bounding Boxes')
        axs[1].imshow(img)
    plt.tight_layout()
    plt.show()


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


if __name__ == "__main__":
    ...