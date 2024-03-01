
from roboflow import Roboflow
import os
import torch
import torch.optim as optim
from pathlib import Path
import ultralytics
from ultralytics import YOLO
import git
from git import Repo
import os
import subprocess

# # Function to run shell commands
# def run_command(command):
#     try:
#         result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
#         print("Command executed successfully:", command)
#         return result.stdout
#     except subprocess.CalledProcessError as e:
#         print("Error executing command:", command)
#         print("Output:", e.stdout)
#         print("Error:", e.stderr)
#         raise
#
# # Clone YOLOv8 repository
# yolov8_repo_url = 'https://github.com/ultralytics/'
# yolov8_repo_path = './'
#
# if not os.path.exists(yolov8_repo_path):
#     print("Cloning YOLOv8 repository...")
#     Repo.clone_from(yolov8_repo_url, yolov8_repo_path)
#     print("YOLOv8 repository cloned.")
# else:
#     print("YOLOv8 repository already exists.")
#
# # Install requirements
# print("Installing YOLOv8 requirements...")
# os.chdir(yolov8_repo_path)  # Change directory to the cloned repo
# run_command('pip install -r requirements.txt')

# Download dataset from Roboflow
print("Downloading dataset from Roboflow...")
from roboflow import Roboflow
rf = Roboflow(api_key="mtJUPQXdun3mtgZUKOK5")
project = rf.workspace("lamar-university-venef").project("shelf-testing")
dataset = project.version(4).download("yolov8")

# After this, continue with your dataset preparation and training code.
# For the training process, refer to the YOLOv8 documentation and adapt the training command accordingly.

model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


def train(yolov8_path, dataset_yaml, img_size=640, batch_size=16, epochs=300, weights_path='yolov8.pt'):
    # Ensure the correct imports for select_device, attempt_load, etc., are here
    device = select_device('cuda' if torch.cuda.is_available() else 'cpu') # Select GPU if available
    model = attempt_load(weights_path, map_location=device)  # Load the YOLOv8 model
    cfg = load_config(dataset_yaml)  # Load dataset configuration
    train_path, val_path, nc, names = cfg['train'], cfg['val'], cfg['nc'], cfg['names']
    train_loader = create_dataloader(train_path, img_size, batch_size, nc, device)[0]
    val_loader = create_dataloader(val_path, img_size, batch_size, nc, device)[0]

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()  # Adjust based on your task

    best_fitness = 0.0
    for epoch in range(epochs):
        model.train()
        for i, (imgs, targets, paths, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}")

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (imgs, targets, paths, _) in enumerate(val_loader):
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss}")

            # Save best model
            if avg_val_loss < best_fitness:
                best_fitness = avg_val_loss
                torch.save(model.state_dict(), f'best_model_{epoch}.pt')
                print(f"Best model saved at epoch {epoch + 1} with Validation Loss: {avg_val_loss}")


# Call the train function
dataset_yaml_path = os.path.join(dataset.location, "data.yaml")

# Call the train function with the appropriate paths and parameters
model.train( dataset_yaml_path, img_size=640, batch_size=16, epochs=300, weights_path='path/to/yolov8.pt')

metrics = model.val()  # evaluate model performance on the validation set