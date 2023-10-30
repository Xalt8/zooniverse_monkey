import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


img_name = "A014_18Z_T15092_MID_x40_z0_i07j10.jpg"
points = {"L":(657.918060302734, 749.88166809082), "N": (738.657089233398, 175.749549865723), "M":(839.444000244141, 454.741058349609)}

img_path = Path() / r"AI Training Sets-20231024T092556Z-001\AI Training Sets" / r"Set4-1_WBC_Images" / img_name



if __name__ =="__main__":
    
    img = cv2.imread(filename=img_path.as_posix()) # <- BGR colour channels 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    ax1.imshow(img)
    ax1.set_title(img_name)
    
    for cell, v in points.items():
        colors = {"L":'red', "N":"blue", "M":"green"}
        ax1.add_patch(plt.Circle(v, radius=30, color=colors[cell], fill=False, linewidth=2))
        text_x = v[0] - 30
        text_y = v[1] - 30
        ax1.text(text_x, text_y, cell, fontsize = 12, color=colors[cell])
    ax1.axis('off')
    
    plt.tight_layout()
    plt.show()
