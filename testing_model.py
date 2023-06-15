from face_detector import YoloDetector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import requests
import torchvision
from pytorch_grad_cam.ablation_cam_multilayer import AblationCAM
from pytorch_grad_cam.eigen_cam import EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image

# test a detection model
model = YoloDetector(target_size=720, device="cpu", min_face=90)
img_path = 'skynews-boris-johnson-texas_6165501.jpg'
orgimg = np.array(Image.open(f"{img_path}"))
bboxes,points = model.predict(orgimg)

# plot the image with bounding boxes and faces detected

# TODO - get the coordinates of the bounding box
# Assuming the coordinates are in the form [x1, y1, x2, y2]
coordinates = bboxes
x1 = coordinates[0][0][0]
y1 = coordinates[0][0][1]
x2 = coordinates[0][0][2]
y2 = coordinates[0][0][3]


def plot_results(img_path):
    # Load the image
    image = Image.open(f"{img_path}")
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)
    # Create a Rectangle patch for the bounding box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()


plot_results(img_path)

# test grad cam
# grad cam
cam = EigenCAM(model,
              target_layers,
              use_cuda=torch.nn.cuda.is_available(),
              reshape_transform=fasterrcnn_reshape_transform)

coco_names = ['person']

def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

image = np.array(Image.open((img_path)))
image_float_np = np.float32(image) / 255
# define the torchvision image transforms
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

input_tensor = transform(image)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = input_tensor.to(device)
# Add a batch dimension:
input_tensor = input_tensor.unsqueeze(0)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval().to(device)

# Run the model and display the detections
boxes, classes, labels, indices = predict(input_tensor, model, device, 0.9)
image = draw_boxes(boxes, labels, classes, image)

# Show the image:
Image.fromarray(image)