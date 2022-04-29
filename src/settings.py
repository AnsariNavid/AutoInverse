import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUMBER_OF_BATCHES_PER_LOG = 100
NUMBER_OF_DECIMAL_PLACES = 5
EPSILON = 1e-10
