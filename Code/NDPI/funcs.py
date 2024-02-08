from pathlib import Path
from openslide import open_slide
import cv2
import math
from tqdm import tqdm
import numpy as np
from ultralytics.engine.results import Results


def save_image_to_path(image: np.ndarray, file_path: str | Path) -> None:
    """ Takes an image array converts it to BGR and saves to disk in RGB
    """
    if isinstance(file_path, Path):
        file_path = file_path.as_posix()
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename=file_path, img=img_bgr)


def make_patches(slide_loc: Path | str,
                 patch_dir: Path,
                 patch_size: tuple[int, int] = (1024, 1024)) -> None:
    """ Creates non-overlapping patched from an image and saves to folder
    """
    assert patch_dir.is_dir(), "Patch folder not found"
    assert slide_loc.is_file(), 'Cannot find slide image file'

    if isinstance(slide_loc, str):
        slide_loc = Path()/slide_loc

    slide_name = slide_loc.parts[-1].split(".")[0]

    using_open_slide: bool = False

    if slide_loc.suffix in ['.ndpi', '.tif', '.tiff']:
        using_open_slide = True
        slide = open_slide(filename=slide_loc)
        width, height = slide.dimensions
    else:
        slide = cv2.imread(filename=slide_loc.as_posix())
        slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)
        width, height = slide.shape[:2]

    assert width > patch_size[1], 'Patch size is bigger than image'
    assert height > patch_size[0], 'Patch size is bigger than image'

    # Calculate the number of patches in each dimension
    num_patches_x = math.ceil(width / patch_size[0])
    num_patches_y = math.ceil(height / patch_size[1])

    for y in tqdm(range(num_patches_y)):
        for x in range(num_patches_x):
            patch_x = x * patch_size[0]
            patch_y = y * patch_size[1]
            if using_open_slide:
                patch = np.array(slide.read_region(
                    (patch_x, patch_y), 0, patch_size).convert('RGB'))
                save_image_to_path(
                    image=patch, file_path=patch_dir/f"{slide_name}_{y}_{x}.jpg")
            else:
                patch = slide[patch_y:patch_y + patch_size[1],
                              patch_x:patch_x + patch_size[0]]
                save_image_to_path(
                    image=patch, file_path=patch_dir/f"{slide_name}_{y}_{x}.jpg")


def get_roi(image: np.ndarray, 
            xy: tuple[int, int] | np.ndarray, 
            offset: tuple[int, int] = (70, 70)) -> np.ndarray | None:
    """ Given an image and xy coordinates, returns the ROI from the image using the offset 
    """
    img_height, img_width, _ = image.shape
    x, y = xy

    # Ensure the ROI does not exceed image dimensions
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))

    # Calculate ROI boundaries using the offset
    roi_x1 = max(0, x - offset[0])
    roi_y1 = max(0, y - offset[1])
    roi_x2 = min(img_width, x + offset[0])
    roi_y2 = min(img_height, y + offset[1])

    # Ensure the ROI is at least 90 pixels in width and height
    if roi_x2 - roi_x1 < 90 or roi_y2 - roi_y1 < 90:
        return None
    
    # Extract the ROI using NumPy array slicing
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2, :]
    return roi


def save_rois(result:Results, folder_path:Path) -> None:
    """Takes a YOLO result object, extracts the ROI boxes 
        and saves them in corresponding folders
    """
    assert isinstance(folder_path, Path), "folder_path needs to be a Path object"
    # No detections
    if result[0].boxes.cpu().data.numpy().size == 0:
        return 
    # For detections
    image_name = Path(result[0].path).parts[-1].split(".")[0]
    print(f"{image_name= }")
    names_dict = result[0].names
    boxes = result[0].boxes.xywh.cpu().numpy().astype(np.int32)
    classification = [names_dict[cls] for cls in result[0].boxes.cls.cpu().numpy().astype(np.uint)]
    for i, (box, cls) in enumerate(zip(boxes, classification)):
        xy = box[:2]
        orig_img = result[0].orig_img[..., ::-1]
        roi = get_roi(image=orig_img, xy=xy)
        if roi is None:
            continue
        print(f'{roi.shape = }')
        box_name = f"B{i:02d}"
        file_name = f"{image_name}_{box_name}.jpg"
        print(f"{file_name= }")
        assert (folder_path/cls).is_dir(), "Class folder not found"
        save_path = folder_path / cls / file_name
        save_image_to_path(image = roi, file_path=save_path)
        print(f'Saved image:{file_name}\nto: {folder_path/cls/file_name}')

if __name__ == "__main__":
    pass
