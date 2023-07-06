from model import YOLOv3, ScalePrediction
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

# Setting the load_model to True
load_model = True
save_model = True
# Defining the model, optimizer, loss function and scaler
model = YOLOv3(num_classes = 20).to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
loss_fn = YOLOLoss()
scaler = torch.cuda.amp.GradScaler()
epochs = 1
batch_size = 1
checkpoint_file = "save model/checkpoint.pth.tar"
if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Loading the checkpoint
    if load_model:
        load_checkpoint(checkpoint_file, model, optimizer, learning_rate)  
    
    
    num_classes = 3
    new_model = YOLOv3(num_classes = 3).to(device)
    state_dict_source = model.state_dict()

    state_dict_target = new_model.state_dict()
    for i, layer_name in enumerate(state_dict_target.keys()):
        if i > 357:
            break
        state_dict_target[layer_name] = state_dict_source[layer_name]
    
    new_model.load_state_dict(state_dict_target)    
    for param in new_model.layers[:10].parameters():
        param.requires_grad = False
    
    load_model = True
    save_model = True
    new_model = YOLOv3(num_classes = 3).to(device)
    new_optimizer = optim.Adam(new_model.parameters(), lr = learning_rate)
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()

    # checkpoint_file = "save model/checkpoint_custom3.pth.tar"
    # if load_model:
    #     load_checkpoint(checkpoint_file, new_model, new_optimizer, learning_rate)  

    # # Defining the train dataset
    epochs = 20
    train_dataset = Dataset(
        csv_file="custom data/train.csv",
        image_dir="custom data/dataset_resized/dataset_resized/Img",
        label_dir="custom data/dataset_resized/dataset_resized/Label",
        anchors=ANCHORS,
        transform=train_transform,
        num_classes = 3
    )

    # # Defining the train data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = 2,
        shuffle = True,
        pin_memory = True,
    )

    # # Scaling the anchors
    scaled_anchors = (
        torch.tensor(ANCHORS) * 
        torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    ).to(device)

    # print(scaled_anchors.shape)

    # # Training the model
    for e in range(1, epochs+1):
        print("Epoch:", e)
        training_loop(train_loader, new_model, new_optimizer, loss_fn, scaler, scaled_anchors)

        # Saving the model
        if save_model:
            save_checkpoint(new_model, new_optimizer, filename=f"save model/checkpoint_custom3.pth.tar")
