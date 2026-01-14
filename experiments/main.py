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
from statistical_analysis import StatisticalAnalysis
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


    tp_gat = TrainingProcessor("SimpleGAT")
    tp_improved = TrainingProcessor("ImprovedGNN")
    tp_simple = TrainingProcessor("SimpleGNN")

    print("\n \n \n SimpleGNN")
    print("___________________________________________________")
    r1= tp_simple.cross_validation_training(data)
    print("\n \n \n ImprovedGNN")
    print("___________________________________________________")
    r2= tp_improved.cross_validation_training(data)
    print("\n \n \n SimpleGAT")
    print("___________________________________________________")
    r3= tp_gat.cross_validation_training(data)
    print("___________________________________________________")
    print("\n \n \n SimpleGNN - balanced")
    print("___________________________________________________")
    r4= tp_simple.cross_validation_training(data, balanced_sampling=True)
    print("\n \n \n ImprovedGNN - balanced")
    print("___________________________________________________")
    r5= tp_improved.cross_validation_training(data, balanced_sampling=True)
    print("\n \n \n SimpleGAT - balanced")
    print("___________________________________________________")
    r6= tp_gat.cross_validation_training(data, balanced_sampling=True)
    print("___________________________________________________")

    r4 = dict(r4)
    r5 = dict(r5)
    r6 = dict(r6)

    r1 = dict(r1)
    r2 = dict(r2)
    r3 = dict(r3)


    d = [r1, r2, r3, r4, r5, r6]

    """
    results = {
                'acc':          [np.float64(0.9846453407510432), np.float64(0.9873157162726007), np.float64(0.9894059209219153), np.float64(0.9866719650307967), np.float64(0.9723266441486192), np.float64(0.9897039539042322), np.float64(0.9871289489370157), np.float64(0.9858414464534075), np.float64(0.9907411086826942), np.float64(0.9886866679912577), np.float64(0.992362408106497), np.float64(0.9902523345916947), np.float64(0.9910033777071329), np.float64(0.9849115835485793), np.float64(0.9898787999205245)], 
                'recall':       [np.float64(0.1971064814814815), np.float64(0.19050925925925927), np.float64(0.27523148148148147), np.float64(0.22094907407407405), np.float64(0.22523148148148148), np.float64(0.21875), np.float64(0.21782407407407406), np.float64(0.21585648148148148), np.float64(0.20613425925925927), np.float64(0.21168981481481483), np.float64(0.29166666666666663), np.float64(0.23032407407407404), np.float64(0.22002314814814813), np.float64(0.3241898148148148), np.float64(0.2033564814814815)], 
                'precision':    [np.float64(0.025514433511202493), np.float64(0.03166649234221749), np.float64(0.052924988150066146), np.float64(0.02800746893759598), np.float64(0.006759252562912493), np.float64(0.030628616062598386), np.float64(0.03623765541352507), np.float64(0.02264386515791885), np.float64(0.031174315063842172), np.float64(0.02504296722417528), np.float64(0.044816427591547206), np.float64(0.03138913070513618), np.float64(0.04605349910208592), np.float64(0.057213780388013634), np.float64(0.04310527881790379)], 
                'f1_score':     [np.float64(0.04252098964271761), np.float64(0.049928510919492144), np.float64(0.08529647033775871), np.float64(0.04624556780118302), np.float64(0.013087230503847141), np.float64(0.051999966978026266), np.float64(0.057624201891724694), np.float64(0.039154713813131604), np.float64(0.05228550005233694), np.float64(0.04263968030990819), np.float64(0.0761264381332433), np.float64(0.053373279602787795), np.float64(0.0683333764301495), np.float64(0.09204170151328676), np.float64(0.06263493670886075)]
            }
    STATISTICAL ANALYSIS
    """
    data = [
    # G1 - Low performance
    {'acc': [0.65, 0.67, 0.66, 0.64, 0.68, 0.65, 0.66, 0.67, 0.65, 0.68, 0.66, 0.64, 0.67, 0.65, 0.66], 
     'precision': [0.62, 0.64, 0.63, 0.61, 0.65, 0.62, 0.63, 0.64, 0.62, 0.65, 0.63, 0.61, 0.64, 0.62, 0.63],
     'recall': [0.58, 0.60, 0.59, 0.57, 0.61, 0.58, 0.59, 0.60, 0.58, 0.61, 0.59, 0.57, 0.60, 0.58, 0.59]},
    
    # G2 - Medium performance  
    {'acc': [0.78, 0.80, 0.79, 0.77, 0.81, 0.78, 0.79, 0.80, 0.78, 0.81, 0.79, 0.77, 0.80, 0.78, 0.79],
     'precision': [0.75, 0.77, 0.76, 0.74, 0.78, 0.75, 0.76, 0.77, 0.75, 0.78, 0.76, 0.74, 0.77, 0.75, 0.76],
     'recall': [0.72, 0.74, 0.73, 0.71, 0.75, 0.72, 0.73, 0.74, 0.72, 0.75, 0.73, 0.71, 0.74, 0.72, 0.73]},
    
    # G3 - High performance
    {'acc': [1.91, 1.93, 1.92, 1.90, 1.94, 1.91, 1.92, 1.93, 1.91, 1.94, 1.92, 1.90, 1.93, 1.91, 1.92],
     'precision': [0.88, 0.90, 0.89, 0.87, 0.91, 0.88, 0.89, 0.90, 0.88, 0.91, 0.89, 0.87, 0.90, 0.88, 0.89],
     'recall': [0.85, 0.87, 0.86, 0.84, 0.88, 0.85, 0.86, 0.87, 0.85, 0.88, 0.86, 0.84, 0.87, 0.85, 0.86]}
    ]

    stats = StatisticalAnalysis(d)


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
