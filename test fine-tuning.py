from model import YOLOv3
from library import *
from config import learning_rate, ANCHORS, device, s
from utils import load_checkpoint, convert_cells_to_bboxes, plot_image, plot_custom_image
from dataset import Dataset
from augment import test_transform
from loss import YOLOLoss
from metrics import nms, mean_average_precision
import multiprocessing
import itertools

# Check point file
checkpoint_file = "save model/checkpoint_custom3.pth.tar"
# Setting the load_model to True
load_model = True
  
# Defining the model, optimizer, loss function and scaler
model = YOLOv3(num_classes=3).to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
loss_fn = YOLOLoss()
scaler = torch.cuda.amp.GradScaler()
multiprocessing.freeze_support()
# Loading the checkpoint
if load_model:
    load_checkpoint(checkpoint_file, model, optimizer, learning_rate)

print("Hoan thanh checkpoint model")

# Defining the test dataset and data loader
test_dataset = Dataset(
    csv_file="custom data/test.csv",
    image_dir="custom data/dataset_resized/dataset_resized/Img",
    label_dir="custom data/dataset_resized/dataset_resized/Label",
    anchors=ANCHORS,
    transform=test_transform,
    num_classes=3
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size = 1,
    num_workers = 2,
    shuffle = True,
)

# # Getting a sample image from the test data loader
# x, y = next(iter(test_loader))
# x = x.to(device)
# i = 0
# model.eval()
# with torch.no_grad():
#     # Getting the model predictions
#     output = model(x)
#     # Getting the bounding boxes from the predictions
#     bboxes = [[] for _ in range(x.shape[0])]
#     anchors = (
#             torch.tensor(ANCHORS)
#                 * torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
#             ).to(device)

#     # Getting bounding boxes for each scale
#     for i in range(3):
#         batch_size, A, S, _, _ = output[i].shape
#         anchor = anchors[i]
#         boxes_scale_i = convert_cells_to_bboxes(
#                             output[i], anchor, s=S, is_predictions=True
#                         )
#         for idx, (box) in enumerate(boxes_scale_i):
#             bboxes[idx] += box
# model.train()
# # Create figure and axes
# # Plotting the image with bounding boxes for each image in the batch
# for i in range(batch_size):
#     # Applying non-max suppression to remove overlapping bounding boxes
#     nms_boxes = nms(bboxes[i], iou_threshold=0.3, threshold=0.6)
#     plot_custom_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)


progress_bar = tqdm(test_loader, leave=True)
  
# Initializing a list to store the losses
pred_boxes = []
target_boxes = []

# Iterating over the training data
for _, (x, y) in enumerate(progress_bar):
    x = x.to(device)
    y0, y1, y2 = (
        y[0].to(device),
        y[1].to(device),
        y[2].to(device),
    )
    temp0 = y[0].reshape(1, -1, 6)
    temp1 = y[1].reshape(1, -1, 6)
    temp2 = y[2].reshape(1, -1, 6)
    target = torch.cat((temp0, temp1, temp2), dim=1)
    with torch.cuda.amp.autocast():
        # Getting the model predictions
        output = model(x)
        # Getting the bounding boxes from the predictions
        bboxes = [[] for _ in range(x.shape[0])]
        anchors = (
                torch.tensor(ANCHORS)
                    * torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
                ).to(device)

        # Getting bounding boxes for each scale
        for i in range(3):
            batch_size, A, S, _, _ = output[i].shape
            anchor = anchors[i]
            boxes_scale_i = convert_cells_to_bboxes(
                                output[i], anchor, s=S, is_predictions=True
                            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
    pred_boxes.append(bboxes)
    target_boxes.append(target)

pred_boxes = np.asarray(pred_boxes)
target_boxes = np.asarray(target_boxes)
pred_boxes = np.squeeze(pred_boxes, axis=1)
target_boxes = np.squeeze(target_boxes, axis=1)

pred = []
for i, x in enumerate(pred_boxes):
    ones_array = np.ones((10647, 1)) * i
    a = np.concatenate((ones_array, x[..., 0:6]), axis = 1).tolist()
    pred.append(a)

target = []
for i, x in enumerate(target_boxes):
    ones_array = np.ones((10647, 1)) * i
    a = np.concatenate((ones_array, x[..., 0:6]), axis = 1).tolist()
    target.append(a)


pred = list(itertools.chain(*pred))
target = list(itertools.chain(*target))

mAp = mean_average_precision(pred, target, num_classes = 3)