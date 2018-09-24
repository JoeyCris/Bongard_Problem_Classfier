import matplotlib.pyplot as plt
import math
import pickle

'''
visualize the general feature map by passing featuer activation matrix 
'''
def visualize_units(units, plt_index):
    #filters = units.shape[2]
    plt.figure(plt_index, figsize=(25,25))
    n_columns = 5
    n_rows = 12 #math.ceil(filters / n_columns) + 1
    for i in range(124, 127):
        plt.subplot(n_rows, n_columns, i-123)
        plt.title('Filter '+str(i))
        plt.imshow(units[:,:,i], interpolation = "nearest", cmap="gray")       
        


# visualize test bongard images feature/activation maps from certain layer
##

for index in range(len(feature_map_conv8[2]['features'])):
    units = feature_map_conv8[2]['features'][index]
    visualize_units(units, index)
