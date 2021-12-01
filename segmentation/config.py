import torch
import os

dataset_dir = "/home/lg08/main/data/standard_collective"
TEST_SPLIT = 0.1
array_file = "extras/solar_array"

SEG_IMAGE_PATH = "/home/lg08/main/data/solar_panel_images/images"
SEG_MASK_PATH = "/home/lg08/main/data/solar_panel_images/masks"

NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 1

INIT_LR = 0.001
NUM_EPOCHS = 40
# NUM_EPOCHS = 3
# BATCH_SIZE = 64
BATCH_SIZE = 4

# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256
# INPUT_IMAGE_WIDTH = 70
# INPUT_IMAGE_HEIGHT = 70

THRESHOLD = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False


# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
