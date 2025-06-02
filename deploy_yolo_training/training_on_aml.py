# --- Import necessary packages ---
from azureml.core import Workspace, Dataset, Run
import os, tarfile, yaml
import torch

# --- Make a temporary directory and mount dataset ---
print("Creating temporary directory...")
mounted_path = './tmp'
os.makedirs(mounted_path, exist_ok=True)
print('Temporary directory made at', mounted_path)

# --- Fetch dataset from AML workspace ---
print("Fetching dataset")
ws = Run.get_context().experiment.workspace
dataset = Dataset.get_by_name(ws, name='mos2_defects')  # ✅ Dataset name in Azure ML
print("Downloading dataset...")
dataset.download(mounted_path, overwrite=True)
print("Downloaded files:", os.listdir(mounted_path))

# --- Extract tar files ---
for file in os.listdir(mounted_path):
    if file.endswith('.tar'):
        print(f"Extracting tar file: {file}")
        tar = tarfile.open(os.path.join(mounted_path, file))
        tar.extractall(path=mounted_path)
        tar.close()

# --- Confirm extraction and paths ---
mos2_defects_folder = os.path.join(mounted_path, "mos2_defects")  # ✅ Make sure this folder exists after untar
print("Contents of extracted folder:", os.listdir(mos2_defects_folder))

# --- Required for OpenCV in container ---
os.system('apt-get install -y python3-opencv')

# --- Clone YOLOv5 repo ---
print("Cloning YOLOv5...")
os.system('git clone https://github.com/ultralytics/yolov5')
print("Contents of current directory:", os.listdir('.'))

# --- Check GPU availability ---
print(f"YOLOv5 environment setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# --- Create YOLOv5 config YAML ---
tag = 'defect'  # ✅ Update tag
tags = [tag]

yolo_yaml = os.path.join('.', 'mos2_defect_detection_yolov5.yaml')
with open(yolo_yaml, 'w') as yamlout:
    yaml.dump(
        {
            'train': os.path.join(mos2_defects_folder, 'train'),
            'val': os.path.join(mos2_defects_folder, 'val'),
            'nc': len(tags),
            'names': tags
        },
        yamlout,
        default_flow_style=None,
        sort_keys=False
    )

# --- Copy config to outputs folder for Azure logs ---
os.makedirs('outputs', exist_ok=True)
os.system('cp ./mos2_defect_detection_yolov5.yaml ./outputs/mos2_defect_detection_yolov5.yaml')

# --- Train the YOLOv5 model ---
os.system('python yolov5/train.py --img 640 --batch 16 --epochs 100 --data ./mos2_defect_detection_yolov5.yaml --weights yolov5s.pt')

# --- Run inference ---
test_image_dir = os.path.join(mos2_defects_folder, 'test', 'images')  # ✅ Make sure this exists
os.system(f'python yolov5/detect.py --weights ./yolov5/runs/train/exp/weights/best.pt --iou 0.05 --save-txt --source {test_image_dir}')

# --- Save outputs for Azure ML tracking ---
os.system('cp -r ./yolov5/runs ./outputs/')
