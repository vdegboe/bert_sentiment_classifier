import torch
import torch.nn as nn
import numpy as np
import os
import random
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=61):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_class_weights(labels_array, num_classes=5):
    weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels_array),
        y=labels_array
    )
    return torch.tensor(weights_array, dtype=torch.float).to(device)


def save_best_model(model, model_path):
    checkpoint_dir = 'checkpoint'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    full_path = os.path.join(checkpoint_dir, model_path)
    torch.save(model.state_dict(), full_path)
    return full_path


