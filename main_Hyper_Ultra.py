from roboflow import Roboflow
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml



# Function to update hyperparameters in YAML file
def update_hyperparameters(hyp_path, lr):
    with open(hyp_path) as f:
        hyp = yaml.safe_load(f)
    hyp['lr0'] = lr  # Update learning rate
    with open(hyp_path, 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Download dataset from Roboflow
print("Downloading dataset from Roboflow...")

rf = Roboflow(api_key="mtJUPQXdun3mtgZUKOK5")
project = rf.workspace("lamar-university-venef").project("shelf-testing")
dataset = project.version(5).download("yolov8")

# Path to your dataset YAML file
dataset_yaml_path = os.path.join(dataset.location, "data.yaml")

# Hyperparameters YAML file path (needs to exist in the YOLOv5 directory or specify your custom path)
hyp_path = os.path.join('./hyp.scratch-high.yaml')

# Define hyperparameters to tune
learning_rates = [0.01, 0.001]
batch_sizes = [16, 32]
epochs = 1  # Keeping epochs fixed for simplicity

# Store results
results = []

for lr in learning_rates:
    for batch in batch_sizes:
        # Update the hyperparameters YAML file for each combination
        update_hyperparameters(hyp_path, lr)

        # Load a pretrained model (recommended for training)
        model = YOLO("yolov8n.pt").to(device)

        # Train the model with current hyperparameters
        print(f"Training with lr={lr}, batch={batch}")
        model.train(data=dataset_yaml_path, imgsz=640, batch=batch, epochs=epochs , optimizer='AdamW')

        # Evaluate the model after training
        metrics = model.val()  # no arguments needed, dataset and settings remembered

        # Access the metrics using the methods provided
        map_50_95 = metrics.box.map  # Mean Average Precision from IoU=0.5 to 0.95
        map_50 = metrics.box.map50  # Mean Average Precision at IoU=0.5
        map_75 = metrics.box.map75  # Mean Average Precision at IoU=0.75
        maps_per_class = metrics.box.maps  # mAP per class

        print(f"Training Results: LR: {lr}, Batch: {batch}, mAP_50-95: {map_50_95}, mAP_50: {map_50}, mAP_75: {map_75}")

        # Store results for comparison
        results.append({
            'lr': lr,
            'batch': batch,
            'mAP_50-95': map_50_95,
            'mAP_50': map_50,
            'mAP_75': map_75
        })

        # Print the results
        print("Results:")
        for res in results:
            print(res)
        # Path: venv/main_Hyper_Ultra.py
        # After evaluating the model
        metrics = model.val('')
        # Inspect the metrics object
        print(dir(metrics))  # List all attributes and methods



# Find the best performing model based on mAP_50-95
best_result = max(results, key=lambda x: x['mAP_50-95'])

# Visualization of Hyperparameter Effects on mAP_50-95
lrs = [result['lr'] for result in results]
batches = [result['batch'] for result in results]
map_50_95s = [result['mAP_50-95'] for result in results]  # Updated to use mAP_50-95 as performance indicator

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each hyperparameter configuration
for i in range(len(results)):
    ax.scatter(lrs[i], batches[i], map_50_95s[i], color='blue')

# Highlight the best result
ax.scatter(best_result['lr'], best_result['batch'], best_result['mAP_50-95'], color='red', label='Best Performance', s=100)

ax.set_xlabel('Learning Rate')
ax.set_ylabel('Batch Size')
ax.set_zlabel('mAP_50-95')
plt.title('Hyperparameter Tuning Results on mAP_50-95')
plt.legend()
plt.show()