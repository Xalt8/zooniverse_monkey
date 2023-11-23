from dataclasses import dataclass

@dataclass
class Cell:
    wbc_class:str
    x_coord: int
    y_coord: int
    center_x_normalized: float
    center_y_normalized: float
    width_normalized: float
    height_normalized: float

    def __post_init__(self):
        if self.wbc_class not in set(["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]):
            raise ValueError("Cell needs to be of types -> [Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil]") 


@dataclass
class ImageAnnotation:
    image_name: str
    image_path: str
    cells: list[Cell] = None
    

    def __repr__(self) -> str:
        if self.cells is not None:
            cell_vals = [[cell.wbc_class, (cell.x_coord, cell.y_coord)] for cell in self.cells]
        else: 
            cell_vals = "No cells"
        return f"Image name: {self.image_name}\nCells:{cell_vals}"


    

if __name__ == "__main__":
    ...