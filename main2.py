from roboflow import Roboflow
import os
from ultralytics import YOLO

# Download dataset from Roboflow
print("Downloading dataset from Roboflow...")
from roboflow import Roboflow
rf = Roboflow(api_key="mtJUPQXdun3mtgZUKOK5")
project = rf.workspace("lamar-university-venef").project("shelf-testing")
dataset = project.version(5).download("yolov8")

# Load a pretrained model (recommended for training)
model = YOLO("yolov8n.pt")

# Path to your dataset YAML file
dataset_yaml_path = os.path.join(dataset.location, "data.yaml")

# Train the model
# Here, you should adjust the path to your dataset YAML file and other training parameters as needed.
model.train(data=dataset_yaml_path, imgsz=640, epochs=25)

# Optionally, evaluate the model after training
metrics = model.val()
