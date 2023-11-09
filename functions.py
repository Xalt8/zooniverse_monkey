from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


def validate_file_path(file_path:str|Path) -> str:
    """ Checks to see if the file path is valid """
    if not isinstance(file_path, (str, Path)):
        raise ValueError(f"{file_path} must be a str or Path object")
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot find this file -> {file_path} ")
    return file_path.as_posix()


def get_image_from_path(file_path:str | Path) -> np.ndarray:
    """ Takes an file path and returns an image as array """
    file_path = validate_file_path(file_path= file_path)
    img = cv2.imread(filename=file_path) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image_to_path(image:np.ndarray, file_path:str|Path) -> None:
    """ Takes an image array converts it to BGR and saves to disk in RGB"""
    if isinstance(file_path, Path):
        file_path = file_path.as_posix()
    
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename = file_path, img = img_bgr)


def get_bad_shapes() -> list[np.ndarray]:
    """ Returns a list with the file names of images that are not in the shape (1000,1000,3) """
    bad_shape = []

    for file_path in (Path() / r"AI Training Sets-20231024T092556Z-001\AI Training Sets\Set4-1_WBC_Images").iterdir():
        img = get_image_from_path(file_path=file_path.as_posix())
        if img.shape == (1000,1000,3):
            continue
        else:
            bad_shape.append(file_path.parts[-1])


    for file_path in (Path() / r"AI Training Sets-20231024T092556Z-001/AI Training Sets/Set5-1_WBC_Images").iterdir():
        img = get_image_from_path(file_path=file_path.as_posix())
        if img.shape == (1000,1000,3):
            continue
        else:
            bad_shape.append(file_path.parts[-1])

    return bad_shape


def get_image_df() -> pd.DataFrame:
    """Returns a dataframe with all the image data """

    set4_df = pd.read_excel(r"AI Training Sets-20231024T092556Z-001\AI Training Sets\Set4-1_TrainingData_20210526.xlsx", index_col=None)
    set4_df = set4_df[set4_df.columns[:-2]]
    set4_df["Animal Date"] = pd.to_datetime(set4_df["Animal Date"])

    set5_df = pd.read_excel(r"AI Training Sets-20231024T092556Z-001\AI Training Sets\Set5-1_TrainingData_20220413.xlsx", index_col=None)
    set5_df = set5_df[set5_df.columns[:-2]]
    set5_df["Animal Date"] = pd.to_datetime(set5_df["Animal Date"])

    df1 = set4_df.iloc[:,:-1]
    df1['set'] = "Set4"
    df2 = set5_df.iloc[:,:-1]
    df2['set'] = "Set5"
    total_df = pd.concat([df1, df2], axis=0)
    total_df = total_df.rename(columns={"X - Coordinate (pixels)": "x_coord", "Y - Coordinate (pixels)": "y_coord", "WBC Classification": "wbc_class"})
    total_df['x_coord'] = total_df['x_coord'].astype(int)
    total_df['y_coord'] = total_df['y_coord'].astype(int) 
    total_df['wbc_class'] = total_df['wbc_class'].astype('category')
    total_df['set'] = total_df['set'].astype('category')

    return total_df


def get_image_path(image_file_name:str, total_df:pd.DataFrame) -> str:
    """ Takes an image name and returns the path to the image in the correct folder """

    image_folder_dict = {'Set5': Path()/ r"AI Training Sets-20231024T092556Z-001\AI Training Sets\Set5-1_WBC_Images",
                         'Set4': Path() / r"AI Training Sets-20231024T092556Z-001\AI Training Sets\Set4-1_WBC_Images"}
    
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


@dataclass
class Cell:
    cell_class:str
    cell_x_coord: int
    cell_y_coord: int
    cell_contour: np.ndarray

    def __post_init__(self):
        if self.cell_class not in set(['N', 'L', 'M','E','B']):
            raise ValueError("Cell needs to be of types -> 'N', 'L', 'M','E','B'") 


@dataclass
class ImageAnnotation:
    image_name: str
    cells: list[Cell] = None

    def __repr__(self) -> str:
        if self.cells is not None:
            cell_vals = [[cell.cell_class, (cell.cell_x_coord, cell.cell_y_coord)] for cell in self.cells]
        else: 
            cell_vals = "No cells"
        return f"Image name: {self.image_name}\nCells:{cell_vals}"


def get_binary_image(image_name: str, total_df:pd.DataFrame) -> np.ndarray:
    """ Takes an image file name applies filters and thresholding 
        and returns an inverted binary image """
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


def get_image_annotation(image_name:str, total_df:pd.DataFrame) -> ImageAnnotation:
    """ Takes an image name a returns a ImageAnnotation object """
    image_annotation = ImageAnnotation(image_name = image_name)
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



if __name__ == "__main__":
    ...