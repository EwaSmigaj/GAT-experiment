import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple
import graph_creation as gc
import cross_validation as cv
from gnn import SimpleGNN, ImprovedGNN
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from collections import defaultdict
from training import TrainingProcessor
import time



def main():
    """Główny punkt wejścia do przetwarzania danych i treningu."""

    """
    GRAPH CREATION
    """

    data = gc.prepare_data()

    """
    TRAINING
    """

    # Main loop

    tp_simple = TrainingProcessor("ImprovedGNN")


    results = tp_simple.cross_validation_training(data)




"""

"""

# print(data)

# model = HeteroGNN(data.metadata(), HIDDEN_CHANNELS, num_edge_features, data).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# class_weights = calculate_class_weights(transactions, device)

# ## 3. Trening i Ewaluacja
# run_training_loop(model, data, optimizer, class_weights)
# final_evaluation(model, data)



if __name__ == '__main__':

    start = time.perf_counter()

    main()

    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")
