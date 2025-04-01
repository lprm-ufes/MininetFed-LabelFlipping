import numpy as np
from functools import reduce
import torch
import pandas as pd

from scipy.stats import shapiro
import scipy.stats as stats

def zscore_robusto(dados):
    mediana = np.median(dados)
    mad = np.median(np.abs(dados - mediana))
    return (dados - mediana) / mad

class FedSketchAgg:
      
    def __init__(self):
      pass
    
    def aggregate(self, all_trainer_samples, weights, metrics, trainer_list):
        """Compute weighted average."""
        num_clients = len(all_trainer_samples)

        dicionario = {}

        for trainer in trainer_list:
            dicionario[trainer] = metrics[trainer]['activation_layer']

        # Empacotar os vetores em uma matriz
        vetores = np.array([metrics[trainer]['activation_layer'] for trainer in trainer_list])

        #Calculating euclidean distance from each trainer to the others
        distancias_totais = []
        for trainer_layer in vetores:
            dist_euclidiana = 0
            for trainer_layer2 in vetores:
                dist_euclidiana += torch.norm(torch.tensor(trainer_layer) - torch.tensor(trainer_layer2))
            dist_euclidiana = dist_euclidiana
            distancias_totais.append(dist_euclidiana.item())

        z_robusto = zscore_robusto(distancias_totais)
        print("ROBUSTO")
        print(z_robusto)

        print("SHAPIRO-WILK")
        print(shapiro(distancias_totais))

        print("Z-SCORE")
        print(stats.zscore(distancias_totais))

        no_selected_trainers = []
        no_selected_trainers_id = []
        for i, item in enumerate(z_robusto):
            if item > 5:
                no_selected_trainers.append(i)
                no_selected_trainers_id.append(trainer_list[i])
            if item < -5:
                no_selected_trainers.append(i)
                no_selected_trainers_id.append(trainer_list[i])
        

        for indice in sorted(no_selected_trainers, reverse=True):
            weights.pop(indice)

        # Compute average weights of each layer
        weights_prime = [
            reduce(np.add, layer_updates) / len(weights)
            for layer_updates in zip(*weights)
        ]
        return weights_prime, no_selected_trainers_id