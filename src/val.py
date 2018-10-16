import data
import model
import numpy as np
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

INPUT_SIZE = (480, 320, 3)

def test_result(y_true, y_pred):
    G = nx.grid_2d_graph(INPUT_SIZE[0], INPUT_SIZE[1])
    for u, v, d in G.edges(data = True):
        d['weight'] =  np.subtract(1.0, np.multiply(2.0, np.abs(y_pred[0,u[0],u[1],0] - y_pred[0,v[0],v[1],0])))

    theta = 0.9

    G.remove_edges_from([(u,v) for (u,v,d) in  G.edges(data=True) if d['weight']<=theta])
    
    components = nx.connected_components(G)
    print("num of comps", len(list(components)))
    labels = dict()
    for (i, comp) in enumerate(components):
        for node in comp:
            labels[node] = i
    viz_segment(label=labels,size_X = 480, size_Y = 320)
    plt.show()
    
        
def viz_segment(label, title = None, size_X = None, size_Y = None):
    I = np.zeros((size_X,size_Y))
    for coord, z in label.items():
        I[coord] = z
    plt.figure(figsize=(4,4))
    plt.imshow(I, cmap = 'prism')
    if title is not None:
        fig = plt.gcf()
        fig.canvas.set_window_title(title)


if __name__ == '__main__':
    print('Init')
    
    model = model.unet()
    model.load_weights("unet.hdf5")
    valdata = data.ValData()
    #result = model.predict(np.expand_dims(valdata[0][99], axis=0),1,verbose=1)
    evaluate = model.evaluate(np.expand_dims(valdata[0][99], axis=0), np.expand_dims(valdata[1][99], axis=0), batch_size=1, verbose=1)
    print("Test loss", evaluate)
    #plt.imshow(np.squeeze(result))
    #plt.show()
    #test_result(np.expand_dims(valdata[1][0], axis=0),result)
    print('Exit')
