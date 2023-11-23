import functions
from pathlib import Path

ROOT_FOLDER = Path(__file__).resolve().parents[1]

if __name__ == "__main__":

    total_df = functions.get_image_df()
    total_df = functions.train_test_split_df(total_df=total_df)
    # yolo_directory = functions.make_yolo_directory(path=ROOT_FOLDER)
    # functions.send_images_annotations(total_df=total_df, yolo_folder_path=yolo_directory)

    print(total_df.head()) 


    