from model import YOLOv3
from library import *
from config import learning_rate, checkpoint_file, ANCHORS, device, batch_size, s
from utils import load_checkpoint, convert_cells_to_bboxes, plot_image, save_checkpoint
from dataset import Dataset
from augment import test_transform
from loss import YOLOLoss
from metrics import nms
import multiprocessing
from augment import train_transform
from train import training_loop
# Taking a sample image and testing the model
  
# Setting the load_model to True
load_model = True
save_model = True
# Defining the model, optimizer, loss function and scaler
model = YOLOv3().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
loss_fn = YOLOLoss()
scaler = torch.cuda.amp.GradScaler()
epochs = 5

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Loading the checkpoint
    if load_model:
        load_checkpoint(checkpoint_file, model, optimizer, learning_rate)  
    
    # Defining the train dataset
    train_dataset = Dataset(
        csv_file="data/PASCAL_VOC/train.csv",
        image_dir="data/PASCAL_VOC/images",
        label_dir="data/PASCAL_VOC/labels",
        anchors=ANCHORS,
        transform=train_transform
    )
    
    # Defining the train data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = 2,
        shuffle = True,
        pin_memory = True,
    )
    
    # Scaling the anchors
    scaled_anchors = (
        torch.tensor(ANCHORS) * 
        torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    ).to(device)

    print(scaled_anchors.shape)
    
    # Training the model
    for e in range(1, epochs+1):
        print("Epoch:", e)
        training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
    
        # Saving the model
        if save_model:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")