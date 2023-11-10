import os

SEP = os.sep

# Dataset
MODELS_DIR = f'.{SEP}src{SEP}models'
DATASET = 'cifar10'

# Model optimization
EPOCHS = 50
MOMENTUM = 0.5
BATCH_SIZE = 100

# Image preprocessing
MEAN, VAR = 0, 0.001
SIGMA = VAR ** 0.5

# Adaptive discriminator augmentation
MAX_TRANSLATION = 0.125
MAX_ROTATION = 0.125
MAX_ZOOM = 0.25
TARGET_ACCURACY = 0.8
STEPS = 1000
