from neural_net import PV_Net
import torch

dataset_dir = "/home/lg08/main/data/standard_collective"
split = 0.9
array_file = "extras/solar_array"

NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 1

INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 64

# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

THRESHOLD = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cnn = PV_Net()
cnn.init_datasets(dataset_dir, split,array_file)
cnn.train()
cnn.save_net()
cnn.load_net()
cnn.test()
cnn.test_whole()
