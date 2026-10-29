"""Configuration for the Mini-VTAB ViT-B/16 BOFT experiments."""

# 768 / 24 = 32 blocks. Six factors expand effective coverage from 24 to 768.
BOFT_BLOCK_SIZE = 24
BOFT_NUM_FACTORS = 6
NUM_EPOCHS = 100
WARMUP_EPOCHS = 10
BOFT_LEARNING_RATE = 1e-4
CLASSIFIER_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
