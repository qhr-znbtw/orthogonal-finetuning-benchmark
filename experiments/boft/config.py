"""Configuration for the ViT-B/16 BOFT experiment."""

NUM_CLASSES = 37

# ViT-B/16 has 768 input features in every Query/Value projection.
# 768 / 24 = 32 blocks and 1 + log2(32) = 6 butterfly factors.
BOFT_BLOCK_SIZE = 24
BOFT_NUM_FACTORS = 6

NUM_EPOCHS = 20
TRAIN_BATCH_SIZE = 32
BOFT_LEARNING_RATE = 1e-4
CLASSIFIER_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
