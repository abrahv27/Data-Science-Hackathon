# --- Import necessary packages ---
from azureml.core import Workspace, Dataset, Run
import os, tempfile, tarfile, yaml

# --- Make a temporary directory and mount dataset ---
print("Creating temporary directory...")
mounted_path = './tmp'
print('Temporary directory made at' + mounted_path)

# --- Fetch dataset from AML workspace ---
print("Fetching dataset")
ws = Run.get_context().experiment.workspace
dataset = Dataset.get_by_name(ws, name='mos2_defects')
print("Downloading dataset...")
dataset.download(mounted_path, overwrite=True)
print("Check that the tar file is there:")
print(os.listdir(mounted_path))
print("defect_image dataset download done")

# --- Extract tar files ---
for file in os.listdir(mounted_path):
    if file.endswith('.tar'):
        print(f"Found tar file: {file}")
        tar = tarfile.open(os.path.join(mounted_path, file))
        tar.extractall()
        tar.close()

files = os.listdir('.')  # Lists all files and directories in the current directory
print(files)

print("")
print("Content of the mos2_defects folder:")
mos2_defects_folder = os.path.join(".","mos2_defects")
print(os.listdir(mos2_defects_folder))

# this is needed for container
os.system('apt-get update')
os.system('apt-get install -y libgl1 python3-opencv --fix-missing')
    
print("Current Directory:")
print(os.getcwd())
print()
    
print("Cloning yolov5")
os.system('git clone https://github.com/ultralytics/yolov5')
print("Check content of '.' folder:")
print(os.listdir('.'))

# Let's check that pytorch recognizes the GPU
import torch
print(f"yolov5 enviroment setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Generate yaml config file for run on Azure GPU
yolo_yaml = os.path.join('.', 'mos2_defect_detection_yolov5.yaml')

tag = 'defect' 
tags = [tag]
with open(yolo_yaml, 'w') as yamlout:
    yaml.dump(
        {'train': os.path.join('../mos2_defects','train'),
        'val': os.path.join('../mos2_defects','val'),
        'nc': len(tags),
        'names': tags},
        yamlout,
        default_flow_style=None,
        sort_keys=False
    )

# --- Copy config to outputs folder for Azure logs ---
os.system('cp ./mos2_defect_detection_yolov5.yaml ./outputs/mos2_defect_detection_yolov5.yaml')

# --- Train the YOLOv5 model ---
os.system('python yolov5/train.py --img 640 --batch 16 --epochs 100 --data ./mos2_defect_detection_yolov5.yaml --weights yolov5s.pt')

# --- Run inference ---
os.system(f'python yolov5/detect.py --weights ./yolov5/runs/train/exp/weights/best.pt --iou 0.05 --save-txt --source ./mos2_defects/Testing')

# --- Save outputs for Azure ML tracking ---
os.system('cp -r ./yolov5/runs ./outputs/')
