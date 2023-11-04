import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas
import time

global runpath
global runtime
runpath=[]
runtime=[]

def tsp(data):  
     uk=geopandas.read_file("uk_regions.geojson")
     uk.plot(figsize=(6,8))
     #world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
     #ax = world[world.name == 'United Kingdom'].plot(
        #color='lightblue', edgecolor='black',figsize=(6,8))
     plt.ylabel('Latitude')
     plt.xlabel('Longitude')
     plt.suptitle('Christofides Algorithm',fontsize=18)
     plt.title('Top 300 cities in UK based on population',fontsize=12)
     start_time=time.time()
    
     for i in range(len(data)): #plot all the vertex
         x=data[i][0]
         y=data[i][1]
         if x==-4.27:
             plt.plot(x,y,'yo',label='Glassgow',markersize=5) #Glassgow
         elif x==-1.91:
             plt.plot(x,y,'go',label='Birmingham',markersize=5) #Birmingham
         elif x==-0.1:
             plt.plot(x,y,'ro',label='London',markersize=5) #lonodn
            #plt.annotate('London', xy=(x,y)) #london
            
         else:
             plt.plot(x,y,'.k',markersize=2.5) #print(x,y)
         plt.legend()
    

    # build a graph
     G = build_graph(data)
    #print("Graph: ", G)

    # build a minimum spanning tree
     MSTree = minimum_spanning_tree(G)
    #print("MSTree: ", MSTree)
    #plot MST
     for i in range(len(MSTree)): #plot all the path of MSTree
         first_node=MSTree[i][0]
         second_node=MSTree[i][1]
         x1=data[first_node][0]
         y1=data[first_node][1]
         x2=data[second_node][0]
         y2=data[second_node][1]
         #plt.plot([x1,x2],[y1,y2],'-k',linewidth=3)
         #print(first_node,second_node)

     # find odd vertexes
     odd_vertexes = find_odd_vertexes(MSTree)
     #print("Odd vertexes in MSTree: ", odd_vertexes)

     # add minimum weight matching edges to MST
     minimum_weight_matching(MSTree, G, odd_vertexes)
     #print("Minimum weight matching: ", MSTree)

     # find an eulerian tour
     eulerian_tour = find_eulerian_tour(MSTree, G)
     for i in range(len(eulerian_tour)-1): #plot all the path of eulerian tour
         first_node=eulerian_tour[i]
         second_node=eulerian_tour[i+1]
         x1=data[first_node][0]
         y1=data[first_node][1]
         x2=data[second_node][0]
         y2=data[second_node][1]
        #plt.plot([x1,x2],[y1,y2],'-k',color='red',linestyle='--')

     #print("Eulerian tour: ", eulerian_tour)

     current = eulerian_tour[0]
     path = [current]
     visited = [False] * len(eulerian_tour)
     visited[eulerian_tour[0]] = True
     length = 0

     for v in eulerian_tour:
         if not visited[v]:
             path.append(v)
             visited[v] = True

             length += G[current][v]
             current = v
            


     length +=G[current][eulerian_tour[0]]
     path.append(eulerian_tour[0])
     for i in range(len(path)-1):
         first_node=path[i]
         second_node=path[i+1]
         x1=data[first_node][0]
         y1=data[first_node][1]
         x2=data[second_node][0]
         y2=data[second_node][1]
         plt.plot([x1,x2],[y1,y2],'-k',color='gray',linewidth=1.0)
         plt.draw()
         plt.pause(0.01)
     #print("Result path: ", path)
     print("Result length of the path: ", length)
     end_time=time.time()
     run_time=end_time-start_time
     print('result run time of finding the path:',run_time)
     runpath.append(length)
     runtime.append(run_time)
     #print(runpath)
     #print(runtime)
    
     #print all the vertex
     '''
     for i in range(len(data)):
        x=data[i][0]
        y=data[i][1]
        #print(x,y)
        plt.plot(x,y,'xg')
     '''
     #plt.plot(label='Top 300 cities in UK based on population')
     plt.pause(1)
     plt.show()
     return length, path


def get_length(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1.0 / 2.0)


def build_graph(data):
    graph = {}
    for this in range(len(data)):
        for another_point in range(len(data)):
            if this != another_point:
                if this not in graph:
                    graph[this] = {}
                    

                graph[this][another_point] = get_length(data[this][0], data[this][1], data[another_point][0],
                                                        data[another_point][1])
                

    return graph


class UnionFind:
    def __init__(self):
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.parents)

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def minimum_spanning_tree(G):
    tree = []
    subtrees = UnionFind()
    for W, u, v in sorted((G[u][v], u, v) for u in G for v in G[u]):
        if subtrees[u] != subtrees[v]:
            tree.append((u, v, W))
            subtrees.union(u, v)
            
    

    return tree


def find_odd_vertexes(MST):
    tmp_g = {}
    vertexes = []
    for edge in MST:
        if edge[0] not in tmp_g:
            tmp_g[edge[0]] = 0

        if edge[1] not in tmp_g:
            tmp_g[edge[1]] = 0

        tmp_g[edge[0]] += 1
        tmp_g[edge[1]] += 1

    for vertex in tmp_g:
        if tmp_g[vertex] % 2 == 1:
            vertexes.append(vertex)

    return vertexes


def minimum_weight_matching(MST, G, odd_vert):
    import random
    random.shuffle(odd_vert)

    while odd_vert:
        v = odd_vert.pop()
        length = float("inf")
        u = 1
        closest = 0
        for u in odd_vert:
            if v != u and G[v][u] < length:
                length = G[v][u]
                closest = u

        MST.append((v, closest, length))
        odd_vert.remove(closest)


def find_eulerian_tour(MatchedMSTree, G):
    # find neigbours
    neighbours = {}
    for edge in MatchedMSTree:
        if edge[0] not in neighbours:
            neighbours[edge[0]] = []

        if edge[1] not in neighbours:
            neighbours[edge[1]] = []

        neighbours[edge[0]].append(edge[1])
        neighbours[edge[1]].append(edge[0])

    # print("Neighbours: ", neighbours)

    # finds the hamiltonian circuit
    start_vertex = MatchedMSTree[0][0]
    EP = [neighbours[start_vertex][0]]

    while len(MatchedMSTree) > 0:
        for i, v in enumerate(EP):
            if len(neighbours[v]) > 0:
                break

        while len(neighbours[v]) > 0:
            w = neighbours[v][0]

            remove_edge_from_matchedMST(MatchedMSTree, v, w)

            del neighbours[v][(neighbours[v].index(w))]
            del neighbours[w][(neighbours[w].index(v))]

            i += 1
            EP.insert(i, w)

            v = w

    return EP


def remove_edge_from_matchedMST(MatchedMST, v1, v2):

    for i, item in enumerate(MatchedMST):
        if (item[0] == v2 and item[1] == v1) or (item[0] == v1 and item[1] == v2):
            del MatchedMST[i]

    return MatchedMST


tsp([[-0.1,51.52], #location of top 300 cities in UK
[-4.27,55.87],
[-1.91,52.48],
[-2.99,53.42],
[-3.22,55.95],
[-1.48,53.39],
[-1.55,53.81],
[-2.6,51.46],
[-2.25,53.48],
[-1.13,52.64],
[-0.36,53.75],
[-1.5,52.42],
[-1.75,53.8],
[-3.18,51.48],
[-1.18,52.97],
[-2.19,53.01],
[-2.15,52.59],
[-4.16,50.38],
[-5.93,54.6],
[-1.5,52.92],
[-0.98,51.45],
[-1.41,50.91],
[-2.08,52.5],
[-2.1,57.15],
[-1.6,55],
[-2.73,53.76],
[-0.44,51.9],
[-1.39,54.91],
[-1.09,50.81],
[-1.97,52.6],
[1.28,52.65],
[-3.96,51.63],
[-1.88,50.73],
[0.71,51.55],
[-1.78,51.57],
[-3,56.47],
[-3.06,53.82],
[-1.98,50.72],
[-0.25,52.59],
[-1.97,52.53],
[-1.23,54.58],
[-2.03,52.51],
[-1.8,53.66],
[-2.43,53.58],
[-2.17,53.42],
[1.14,52.07],
[-4,55.79],
[-1.11,53.96],
[-1.35,53.44],
[-1.26,51.76],
[-0.15,50.83],
[-2.24,51.86],
[-0.4,51.66],
[-3,51.59],
[-0.61,51.52],
[0.46,51.57],
[-1.84,52.56],
[-2.49,53.75],
[-2.75,53.46],
[-0.57,51.32],
[-0.57,51.32],
[-0.9,52.24],
[0.47,51.73],
[0.9,51.88],
[0.13,52.21],
[0.27,50.78],
[-0.38,50.82],
[-3.54,50.73],
[0.55,51.39],
[-1.78,52.42],
[-1.88,53.73],
[-2.09,51.9],
[0.51,51.28],
[-2.16,53.62],
[-0.2,51.13],
[-3.01,53.66],
[-3.79,56.02],
[-3.04,53.39],
[-0.11,53.57],
[-1.57,54.52],
[-2.24,52.2],
[-2.38,51.39],
[-1.22,54.69],
[-2.64,53.55],
[0.57,50.86],
[-2.6,53.39],
[-7.33,55],
[-1.43,55],
[-1.32,54.57],
[-0.34,51.75],
[-0.55,53.25],
[-2.92,53.2],
[-1.11,51.27],
[-1.62,54.97],
[-1.94,52.31],
[-0.48,51.76],
[-0.21,51.92],
[-0.67,53.59],
[-1.49,53.57],
[0.51,51.38],
[-2.23,53.02],
[-2.26,53.8],
[-1.5,53.68],
[-0.78,51.64],
[-2.98,51.36],
[-1.05,50.89],
[-0.49,52.15],
[0.11,51.78],
[-2.95,54.91],
[-4.79,55.96],
[-4.18,55.77],
[-1.21,53.15],
[-1.42,53.25],
[-1.69,52.63],
[-1.12,53.53],
[-1.49,52.53],
[-0.18,50.84],
[-2.31,53.48],
[-0.58,51.24],
[-1.22,52.93],
[-2.75,52.71],
[-1.54,53.99],
[-1.14,50.8],
[-4.11,55.79],
[-2.45,53.08],
[-0.77,51.42],
[1.72,52.49],
[-2.75,53.34],
[-2.91,53.29],
[-0.29,51.33],
[-3.01,53.47],
[-2.31,53.6],
[-2.53,51.46],
[-2.04,52.7],
[-2.12,52.81],
[-1.28,52.37],
[-3.52,50.47],
[-4.64,55.47],
[0.26,51.14],
[-1.65,52.81],
[-0.83,51.83],
[-1.53,54.91],
[-3.05,53.43],
[-0.75,52.03],
[-0.69,50.79],
[-3.11,51.03],
[-0.74,51.53],
[-3.01,53.35],
[0.86,51.15],
[1.4,51.39],
[0.22,51.44],
[-2.24,53.4],
[-2.06,52.46],
[-2.75,53.36],
[-5.68,54.66],
[-5.93,54.66],
[-2.73,52.06],
[1.71,52.59],
[-2.27,52.4],
[-3.12,53.39],
[-2.16,52.46],
[-2.34,53.42],
[-3.46,51.63],
[-1.56,52.29],
[-0.76,51.3],
[-1.19,50.85],
[-3.29,51.4],
[-2.86,53.41],
[-0.47,51.37],
[-2.48,50.62],
[-0.03,51.71],
[-3.5,55.91],
[-0.54,50.82],
[-3.03,53.5],
[0.32,51.48],
[-2.86,54.08],
[-2.15,53.27],
[-0.51,51.43],
[0.39,51.44],
[-0.77,51.26],
[-0.77,51.26],
[-1.6,53.74],
[-1.64,53.72],
[-0.54,51.89],
[-1.92,53.87],
[-0.75,51.33],
[-0.73,52.4],
[0.55,51.56],
[-0.73,52.5],
[0.31,51.63],
[-3.22,54.13],
[-2.8,54.05],
[-1.08,52.98],
[-1.22,52.77],
[1.14,51.8],
[-3.16,57.14],
[-1.27,52.9],
[-0.22,51.24],
[1.16,51.1],
[-0.36,51.38],
[-0.33,51.07],
[-3.58,50.44],
[-0.85,51.42],
[-3.04,51.65],
[-1.53,54.99],
[-1.37,52.54],
[0.4,52.75],
[-0.74,52],
[-0.7,52.3],
[-3.94,56.12],
[-3.83,51.65],
[-4.57,55.95],
[-4.17,51.68],
[-1.35,50.97],
[-2.11,53.5],
[-0.99,50.86],
[-2.53,53.5],
[-2.33,53.52],
[-0.21,51.8],
[-4.5,55.62],
[-0.34,51.3],
[-6.06,54.52],
[-2.47,52.68],
[0.46,50.85],
[-3.47,56.4],
[-4.24,57.49],
[-2.65,53.39],
[-1.73,54.98],
[-3.99,55.96],
[-1.35,52.06],
[-3.02,53.75],
[-2.8,53.55],
[0.73,51.35],
[-1.29,51.68],
[-2.21,53.56],
[-2.65,50.95],
[-2.31,53.46],
[-1.28,50.87],
[-3,53.05],
[-0.43,54.29],
[-0.43,54.29],
[-2.36,53.38],
[-1.13,52.93],
[-1.8,51.08],
[-2.38,53.45],
[1.07,51.29],
[-3.58,51.51],
[0.06,51.66],
[-1.56,52.78],
[-1.13,53.31],
[-1.76,50.73],
[-1.71,54.87],
[-2.23,51.75],
[-3.38,51.76],
[-0.22,51.32],
[-2.12,53.38],
[-1.49,51.21],
[1.4,51.34],
[-3.01,51.13],
[-2.4,53.52],
[-1.33,51.07],
[-3.46,56.07],
[-1.35,53.72],
[-2.7,53.7],
[-2.12,53.45],
[-0.8,51.21],
[-1.13,53.01],
[0.63,51.52],
[-2.32,53.57],
[-5.3,50.22],
[1.11,51.37],
[-3.06,55.89],
[-2.42,51.54],
[-1.14,54.57],
[-0.82,53.08],
[-1.28,54.61],
[-4.69,55.62],
[-1.27,53.13],
[-2.39,53.76],
[-2.89,53.49],
[-1.6,54.78],
[-1.07,54.61],
[0.42,51.63],
[-2.52,53.26],
[-3.84,51.62],
[-0.02,51.76],
[-0.65,52.91],
[-1.31,52.97],
[-0.67,51.93],
[-1.52,55.12],
[-1.45,55.01],
[-1.58,54.86],
[0.57,51.88],
[-2.81,53.43],
[-0.02,52.99],
[-1.47,54.85],
[-1.33,51.4],
[-2.33,53.49],
[-0.28,51.96],
[-3.04,51.7]])
