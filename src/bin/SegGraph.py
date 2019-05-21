import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import itertools#some function definitions

DELTA_TOLERANCE = 1.0e-12

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def minimum_energy(G,print_all_graphs = False):
    min_energy = 0
    M = G.copy()
    for cut_edges in powerset(G.edges):
        H = G.copy()
        H.remove_edges_from(cut_edges)
        components = [[x for x in comp] for comp in nx.connected_components(H)]
        energy = 0
        for edge in cut_edges:
            if not any(edge[0] in comp and edge[1] in comp for comp in components):
                energy+=G.edges[edge]['weight']
        
        if print_all_graphs:
            drawing(H)
            print('energy:', energy)
            
        if energy < min_energy:
            min_energy = energy
            M = H.copy()
    #print('min energy: ', min_energy)
    #drawing(M)
    return(min_energy, M)

from itertools import islice
def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight), k))

from itertools import tee
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def FindFrustrated(G):

    for u, v, w in G.edges(data = 'weight'):
        posPath = 0 
        negPath = 0 
        for path in nx.shortest_simple_paths(G, u, v):            
            pathWeights = [G[edge[0]][edge[1]]['weight'] for edge in pairwise(path)]
            maxp = max(pathWeights)
            minp = min(pathWeights)
            if maxp > 0 and minp > 0: 
                posPath = posPath + 1
                
        if posPath > 0 and w < 0:
            return True            
        
    return False

def GetLabelsAtThreshold(G,theta=0):
    lg = G.copy()    
    lg.remove_edges_from([(u,v) for (u,v,d) in  G.edges(data=True) if d['weight']<theta])
    L = {node:color for color,comp in enumerate(nx.connected_components(lg)) for node in comp}    
    return L

def GetWatershedGraph(G):
    WG = G.copy()    
    #this function returns the graph WG with new weights of max(min(neighbors))    
    for (u,v,d) in WG.edges(data = True):
        #print(u)        
        uew = [WG[u][ues]['weight'] for ues in WG[u] if ues != v]
        vew = [WG[v][ves]['weight'] for ves in WG[v] if ves != u]        
        d['weight'] = d['weight'] - min(max(uew), max(vew))
    return WG

def GetLabelEnergy(G, L):
    E = 0
    for (u,v,d) in G.edges(data = True):
        if L[u] != L[v]:
            E = E + d['weight']    
    return E


def GetLowerBound(WG):    
    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]
    totalNeg = sum([d[2] for d in edgeWeights if d[2] < 0])

    return totalNeg


def FindMinEnergyThreshold(WG, eval=None):
    
    mySets = nx.utils.UnionFind()   
    nextNode = dict()

    for n in WG:
        nextNode[n] = mySets[n]

    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    

    if eval is None:
        evalWeights = edgeWeights
        eval = WG
    else:
        evalWeights = [(u,v,w) for (u,v,w) in eval.edges(data = 'weight')]    

    totalPos = sum([d[2] for d in evalWeights if d[2] > 0])
    totalNeg = sum([d[2] for d in evalWeights if d[2] < 0])
    accTotal = [0]*len(evalWeights)

    accTotal[0] = totalPos + totalNeg
    #print("Energy 0: " + str(accTotal[0]) + " from Pos: " + str(totalPos) + ", Neg: " + str(totalNeg))

    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 
    
    ei = 1      # edge index
    lowE = accTotal[0]
    lowT = sortedEdges[0][2] + DELTA_TOLERANCE

    for u, v, w in sortedEdges:
        if mySets[u] != mySets[v]:
            accWeight = 0.0
            # traverse nodes in u and look at edges
            # if fully connected we should probably traverse nodes u and v instead
            done = False
            cu = u
            while not done:
                for uev in WG[cu]:                
                    if mySets[uev] == mySets[v]:
                        accWeight = accWeight + eval[cu][uev]['weight']
                cu = nextNode[cu]
                if cu == u:
                    done = True

            # Merge sets
            mySets.union(u, v)
            # Swap next pointers... this incrementally builds a pointer cycle around all the nodes in the component
            tempNext = nextNode[u]
            nextNode[u] = nextNode[v]
            nextNode[v] = tempNext

            accTotal[ei] = accTotal[ei-1] - accWeight            
            #print("Energy at threshold " + str(w) + ": " + str(accTotal[ei]))
            if accTotal[ei] < lowE:
                lowE = accTotal[ei]
                lowT = w - DELTA_TOLERANCE


            ei = ei + 1
        
    #print("Lowest Energy: " + str(lowE) + " at threshold " + str(lowThreshold)) 
    return(lowT, lowE)

#----------------------------

def GentleThreshold(G):
    #find the bigest threshold that will split the graph
    lg = G.copy()
    
    for t in sorted(set([w for (u,v,w) in G.edges(data = 'weight')]),reverse = True):
        lg.remove_edges_from([(u,v) for (u,v,d) in G.edges(data=True) if d['weight']<t])
        if nx.number_connected_components(lg) < 50:
            break
        else:
            lg = G.copy()
    
    label = {node:color for color,comp in enumerate(nx.connected_components(lg)) for node in comp}
    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(lg)]
    return label, subgraphs


def MultiGentleThreshold(G, subgraphs):
    #subgraphs have unique nodes from original graph G
    max_label = 0
    label = {}
    S_comp = []
    for graph in subgraphs:
        L,S = GentleThreshold(graph)
        L.update((k,v+max_label+1) for k,v in L.items())
        label = {**label, **L}
        max_label = max(list(label.values()))
        S_comp.extend(S)
    return label, S_comp

def MultiMultiGentleThreshold(G, steps = None):
    if steps == None:
        steps = 5

    subgraphs = [G]
    labels = []

    for i in range(steps):
        label, subgraphs = MultiGentleThreshold(G, subgraphs)
        labels.append(label)
    return labels


#find out if two labels is connected
#inputs: the graph G, 2 segments (2 integers)
def segments_is_connected(G, labels, x, y):
    nodes_x = [node for node,color in labels.items() if color == x]
    nodes_y = [node for node,color in labels.items() if color == y]
    return any([G.has_edge(u,v) for u,v in itertools.product(nodes_x,nodes_y)])

#spanning forest
def spanning_forest(G):
    print('spanning_forest....')
    H = nx.Graph()
    H.add_nodes_from(G)
    for u,v,w in sorted(G.edges(data='weight'), key=lambda x:x[2], reverse=True):
        if not(H.has_node(u) or H.has_node(v)):
            H.add_edge(u, v, weight = w)
    #label = {node:color for color,comp in enumerate(nx.connected_components(H)) for node in comp}
    #subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(H)]
    return H

def spanning_forest_2(G, H):
    print('spanning_forest_2....')
    label = {node:color for color,comp in enumerate(nx.connected_components(H)) for node in comp}
    seen_label = set()
    for u,v,w in sorted(G.edges(data='weight'), key=lambda x:x[2], reverse=True):
        if label[u] != label[v] and all([label[u] not in seen_label, label[v] not in seen_label]):
            H.add_edge(u, v, weight = w)
            seen_label.update([label[u], label[v]])
    label = {node:color for color,comp in enumerate(nx.connected_components(H)) for node in comp} 
    print('num of classes: ', max(label.values())+1)
    return H

def multi_spanning(G, steps=None):
    H = spanning_forest(G)
    label = {node:color for color,comp in enumerate(nx.connected_components(H)) for node in comp}
    labels = [label]
    if steps:
        for i in range(steps - 1):
            H = spanning_forest_2(G,H)
            cut_in_each_segment(H)
            label = {node:color for color,comp in enumerate(nx.connected_components(H)) for node in comp}
            labels.append(label)
    else:
        while max(label.values()) > 50:
            H = spanning_forest_2(G,H)
            cut_in_each_segment(H)
            label = {node:color for color,comp in enumerate(nx.connected_components(H)) for node in comp}
            labels.append(label)
    return labels

def cut_in_each_segment(H):
    for c in nx.connected_components(H):
        if len(c) > 30:
            segG = H.subgraph(c)
            u,v,w = min(segG.edges(data='weight'), key=lambda x:x[2])
            H.remove_edge(u,v)
    return H