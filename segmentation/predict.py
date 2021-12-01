# import the necessary packages
import config
from unet import UNet
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

def prepare_plot(origImage, origMask, predMask, count):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)

    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")

    # set the layout of the figure and display it
    figure.tight_layout()
    # figure.show()
    figure.savefig(f'plots/plot{count}.png', dpi=300, bbox_inches='tight')


def make_predictions(model, imagePath, count):
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        # resize the image and make a copy of it for visualization
        # image = cv2.resize(image, (128, 128))
        image = cv2.resize(image, (256, 256))
        orig = image.copy()
        print("-----------------------------")
        print(orig)
        print(orig.dtype)
        print("-----------------------------")

        # find the filename and generate the path to ground truth
        # mask
        filename = imagePath.split(os.path.sep)[-1]
        filename = filename.split(".")
        filename = filename[0] + "_label." + filename[1]
        print("filename")
        print(filename)
        groundTruthPath = os.path.join(config.SEG_MASK_PATH,
            filename)

        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gtMask = cv2.imread(groundTruthPath, 0)
        print("gt mask type")
        print(gtMask.dtype)
        gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)

        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()

        # filter out the weak predictions and convert them to integers
        predMask = (predMask > config.THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)
        print("pred mask type")
        print(predMask.dtype)

        # prepare a plot for visualization
        prepare_plot(orig, gtMask, predMask, count)


# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
# unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
unet = UNet().to(config.DEVICE)
unet.load_state_dict(torch.load(config.MODEL_PATH))
unet.eval()
# unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
print("got here")

# iterate over the randomly selected test image paths
count = 0
print(count)
for path in imagePaths:
    print(count)
    # make predictions and visualize the results
    make_predictions(unet, path, count)
    count = count + 1
