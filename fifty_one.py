import fiftyone as fo

DATASET_PATH = '/home/aubrey/Desktop/Guam07-merged_results/YOLO-prepped'

# Define the path to your YOLOv11 dataset
dataset_dir = DATASET_PATH

# Define a name for your FiftyOne dataset
dataset_name = 'Guam07-2025-10-09'

# Load the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    # name=dataset_name,
)

# Optional: View the dataset in the FiftyOne App
session = fo.launch_app(dataset)
session.wait()