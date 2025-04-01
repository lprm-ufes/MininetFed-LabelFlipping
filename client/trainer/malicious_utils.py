import numpy as np

def MNIST_label_flipping(dataset):
    for i in range(len(dataset.targets)):            
        rand = dataset.targets[i]
        
        while(rand == dataset.targets[i]):
            rand = np.random.randint(0, 9)

        dataset.targets[i] = rand

def FMNIST_label_flipping(dataset):
    for i in range(len(dataset.targets)):            
        rand = dataset.targets[i]
        
        while(rand == dataset.targets[i]):
            rand = np.random.randint(0, 9)

        dataset.targets[i] = rand
    