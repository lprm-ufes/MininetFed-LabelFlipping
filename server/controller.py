import random
import numpy as np
import pandas as pd
from clientSelection import ClientSelection
from aggregator import Aggregator


class Controller:
    def __init__(self, min_trainers=2, trainers_per_round=2, num_rounds=5):
        self.trainer_list = []
        self.min_trainers = min_trainers
        self.trainers_per_round = trainers_per_round
        self.current_round = 0
        self.num_rounds = num_rounds # total number of rounds
        self.num_responses = 0 # number of responses received on aggWeights and metrics
        self.weights = [] # save weights for agg
        self.trainer_samples = [] # save num_samples scale for agg
        self.id_round = []
        self.acc_list = []
        self.mean_acc_per_round = []
        self.no_selected_clients = []
        self.acc_list_order = []
        self.clientSelection = ClientSelection()
        self.aggregator = Aggregator()
        self.metrics={}
    
    # getters
    def get_trainer_list(self):
        return self.trainer_list
    
    def get_current_round(self):
        return self.current_round
    
    def get_num_trainers(self):
        return len(self.trainer_list)
    
    def get_num_responses(self):
        return self.num_responses

    def get_mean_acc(self):
        mean = 0

        for i, trainer in enumerate(self.acc_list_order):
            if trainer in self.no_selected_clients:
                self.acc_list.pop(i)

        for item in self.acc_list:
            mean += item
        mean = mean / len(self.acc_list)
        
        print("MEAN")
        print(self.acc_list)
        print(self.acc_list_order)
        print(self.no_selected_clients)
        self.mean_acc_per_round.append(mean)  #save mean acc
        return mean

        #mean = float(np.mean(np.array(self.acc_list)))
        #self.mean_acc_per_round.append(mean)  #save mean acc
        #return mean
    
    # "setters"
    def update_metrics(self, trainer_id, metrics):
        self.metrics[trainer_id] = metrics
    
    def update_num_responses(self):
        self.num_responses += 1
    
    def reset_num_responses(self):
        self.num_responses = 0
    
    def reset_acc_list(self):
        self.acc_list = []
        self.acc_list_order = []
    
    def update_current_round(self):
        self.current_round += 1
    
    def add_trainer(self, trainer_id):
        self.trainer_list.append(trainer_id)

    def add_weight(self, weights):
        self.weights.append(weights)

    def add_id(self, id):
        self.id_round.append(id)
    
    def add_num_samples(self, num_samples):
        self.trainer_samples.append(num_samples)
    
    def add_accuracy(self, acc, id):
        self.acc_list.append(acc)
        self.acc_list_order.append(id)

    # operations
    
    def select_trainers_for_round(self):
        if self.current_round <= 1:
            #list = self.clientSelection.select_trainers_for_round_initial(self.trainer_list, self.metrics)
            list = self.clientSelection.select_trainers_for_round(self.trainer_list, self.metrics)
            print(list)
            return list
        else:
            #list = self.clientSelection.select_trainers_for_round_initial(self.trainer_list, self.metrics)
            list = self.clientSelection.select_trainers_for_round(self.trainer_list, self.metrics)
            print(list)
            return list
            

    
    def agg_weights(self):
        self.no_selected_clients = []
        agg_weights = self.aggregator.aggregate(self.trainer_samples, self.weights)
        #agg_weights, self.no_selected_clients = self.aggregator.aggregate(self.trainer_samples, self.weights, self.metrics, self.id_round)
        #agg_weights= self.aggregator.aggregate(self.trainer_samples, self.weights)

        # reset weights and samples for next round
        self.weights = []
        self.trainer_samples = []
        self.id_round = []

        return agg_weights