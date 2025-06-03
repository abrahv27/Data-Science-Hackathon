# import necessary packages
from azureml.core import Workspace, Dataset, Run
import os, tempfile, tarfile, yaml

# Make a temporary directory and mount dataset
print("Creating temporary directory...")
mounted_path = './tmp'
os.makedirs(mounted_path, exist_ok=True)
print('Temporary directory created at:', mounted_path)

# Get the dataset from the current workspace, and download it
print("Fetching dataset from Azure ML workspace...")
ws = Run.get_context().experiment.workspace
dataset = Dataset.get_by_name(ws, name='mos2_defects')
print("Downloading dataset...")
dataset.download(mounted_path, overwrite=True)

print("Contents of ./tmp after download:")
print(os.listdir(mounted_path))

# Untar all files in the mounted directory
for file in os.listdir(mounted_path):
    if file.endswith('.tar'):
        print(f"Extracting tar file: {file}")
        tar = tarfile.open(os.path.join(mounted_path, file))
        tar.extractall(path=mounted_path)
        tar.close()

# Dynamically detect extracted folder name
extracted_dirs = [
    f for f in os.listdir(mounted_path)
    if os.path.isdir(os.path.join(mounted_path, f)) and not f.startswith('.')
]

if not extracted_dirs:
    raise Exception("No dataset directory found in ./tmp after extraction.")

dataset_folder = os.path.join(mounted_path, extracted_dirs[0])

print("Detected dataset folder:", dataset_folder)
print("Contents:", os.listdir(dataset_folder))

# Install OpenCV (needed in container)
os.system('apt-get install -y python3-opencv')

print("Current working directory:")
print(os.getcwd())
print()

# Clone YOLOv5
print("Cloning YOLOv5 GitHub repository...")
os.system('git clone https://github.com/ultralytics/yolov5')
print("Contents of current directory after clone:")
print(os.listdir('.'))

# Check PyTorch GPU availability
import torch
device_name = torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'
print(f"YOLOv5 environment setup complete. Using torch {torch.__version__} ({device_name})")

# Generate YOLOv5 YAML config
tag = 'defect'  # <-- use appropriate class name
tags = [tag]
yolo_yaml = os.path.join('.', 'mos2_defect_detection_yolov5.yaml')

with open(yolo_yaml, 'w') as yamlout:
    yaml.dump(
        {
            'train': os.path.join(dataset_folder, 'train'),
            'val': os.path.join(dataset_folder, 'val'),
            'nc': len(tags),
            'names': tags
        },
        yamlout,
        default_flow_style=None,
        sort_keys=False
    )

# Copy YAML to outputs for tracking
os.makedirs('./outputs', exist_ok=True)
os.system(f'cp {yolo_yaml} ./outputs/')

# Train YOLOv5 model
train_cmd = f'python yolov5/train.py --img 640 --batch 16 --epochs 100 --data {yolo_yaml} --weights yolov5s.pt'
print(f"Running training: {train_cmd}")
os.system(train_cmd)

# Run detection ONLY if best.pt exists
weights_path = './yolov5/runs/train/exp/weights/best.pt'
if os.path.exists(weights_path):
    detect_cmd = f'python yolov5/detect.py --weights {weights_path} --iou 0.05 --save-txt --source {os.path.join(dataset_folder, "test/images")}'
    print(f"Running detection: {detect_cmd}")
    os.system(detect_cmd)
    os.system('cp -r ./yolov5/runs ./outputs/')
else:
    print("❌ Skipping detection — weights file best.pt not found.")
