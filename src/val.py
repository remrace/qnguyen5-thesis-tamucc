import data
import model
import numpy as np
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
import SegEval
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

INPUT_SIZE = (480, 320, 3)

def test():
    my_model = model.unet()
    my_model.load_weights("unet.hdf5")
    ID = '25098'
    t = data.Sample('train', ID = ID)
    image = np.expand_dims(t[0], axis = 0)
    nlabel = np.expand_dims(t[1], axis = 0)
    elabel = t[2]
    print(image.shape, nlabel.shape)
    result = my_model.predict([image, nlabel], batch_size=1)
    #print(result)
    
    G = nx.grid_2d_graph(INPUT_SIZE[0], INPUT_SIZE[1])
    nlabels_dict = dict()
    correlation_clustering_error = 0.0
    for u, v, d in G.edges(data = True):
        if u[0] == v[0]:
            channel = 0
        else:
            channel = 1
        d['weight'] =  result[0, u[0], u[1], channel]

        nlabels_dict[u] = nlabel[0, u[0], u[1], 0]
        nlabels_dict[v] = nlabel[0, v[0], v[1], 0]

        if d['weight'] * elabel[u[0], u[1], channel] < 0.0:
            correlation_clustering_error += abs(d['weight'])
    theta = 0.0
    G.remove_edges_from([(u,v) for (u,v,d) in  G.edges(data=True) if d['weight']<=theta])
    
    pred_label_dict = dict()
    for s, comp in enumerate(nx.connected_components(G)):
        #print(s, len(comp))
        for node in comp:
            pred_label_dict[node] = s
        #print(s, comp)


    print('num of segs: ', len(list(nx.connected_components(G))))
    print('rand error: ', SegEval.FindRandErrorAtThreshold(WG=G, gtLabels=nlabels_dict, T=0.0))
    print('correlation_clustering_error: ', correlation_clustering_error/np.sum(np.abs(result)))

    
    result_image = np.zeros((480, 320))

    for node, label in pred_label_dict.items():
        result_image[node] = label
        #print(label)

    fig=plt.figure(figsize=(8, 4))

    fig.add_subplot(1, 3, 1)
    plt.imshow(np.squeeze(t[1]), cmap='nipy_spectral')
    fig.add_subplot(1, 3, 2)
    plt.imshow(result_image, cmap='nipy_spectral')
    
    plt.show()
    



if __name__ == '__main__':
    print('Init')
    test()
    print('Exit')
