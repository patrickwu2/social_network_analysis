import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Network(object):
    """docstring for Network"""
    def __init__(self, graph, weight_mode, broadcast_prob, seed):
        super(Network, self).__init__()
        # set random seed
        if seed != None:
            random.seed(seed)
        self.weight_mode = weight_mode
        self.broadcast_prob = broadcast_prob
        # init graph, node, pos
        self.G = nx.Graph()
        self.G.add_nodes_from(graph.keys())
        self.node = list(self.G.nodes())
        self.pos=nx.spring_layout(self.G, random_state=seed)
        # init everyone don't know gossip
        self.gossip = {}
        for x in self.node:
            self.gossip.update({x:False})
        nx.set_node_attributes(self.G, self.gossip, 'gossip')
        # init experimental setting
        for k, v in graph.items():
            if self.weight_mode == "uniform":
                self.G.add_edges_from(([(k, t) for t in v]), weight=np.random.random())
            elif self.weight_mode == "normal":
                self.G.add_edges_from(([(k, t) for t in v]), weight=np.random.normal(loc=0.5, scale=0.1))

        '''
        Graph
            Input:
                graph: a dictionary with un-directed graph

            Attribute:
                Node: self.G[<node>][<node attribute>] (assign or get value)
                Edge: self.G[<node1>][<node2>][<edge attribute>] (assign or get value)
                Neighbor: self.G.neighbors(<node>)

            To do list:
                1. differnet init graph (including diff social network, diff random weight, diff relation update)
                3. Beautify the graph 
        '''

        
    def init_round(self):
        # everyone not know
        for x in self.G.nodes():
            self.G.nodes[x]['gossip'] = False
        # the strongest link has the highest probability
        link = []
        weight = []
        for v, u in self.G.edges():
            link.append((v, u))
            weight.append(self.G[v][u]['weight'])
        gossip_event = random.choices(link, weight)[0]
        origin, gossiper = gossip_event if random.random() > 0.5 else gossip_event[::-1]
        self.G.node[origin]['gossip'] = True
        # pick victim
        if self.G.degree[origin] == 2:
            return origin, gossiper, None
        victim = random.choice(list(nx.common_neighbors(self.G, origin, gossiper)))     
        return origin, gossiper, victim

    def get_gossiper(self, origin, victim):
        # prob to broadcast gossip or not
        r = random.random()
        if r >= self.broadcast_prob:
            return None

        # find all neighbors
        gossiper = []
        w = []
        for x in self.G.neighbors(origin):
            if victim in self.G.neighbors(x):
                gossiper.append(x)
                w.append(self.G[x][origin]['weight'])
        try: # has neighbor
            o_next = random.choices(gossiper, w)[0]
            if self.G.node[o_next]['gossip'] == False:
                self.G.node[o_next]['gossip'] = True
                return o_next
            else: # no unknown neighbor
                return None           
        except: # no neighbor
            return None


    def update_relation(self, origin, victim, gossiper):
        self.G[origin][victim]['weight'] = self.G[origin][victim]['weight'] ** 2
        self.G[gossiper][victim]['weight'] = self.G[gossiper][victim]['weight'] ** 2
        self.G[origin][gossiper]['weight'] = self.G[origin][gossiper]['weight'] ** (0.5)
        
    def draw(self, iteration):
        # weight x width
        weights = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        width = [x * 7 for x in weights]
        # gossip color
        color = []
        for x in self.G.nodes():
            if self.G.nodes[x]['gossip'] == True:
                color.append('red')
            else:
                color.append('green')

        # draw
        nx.draw_networkx_nodes(self.G,self.pos,node_size=300, node_color=color)
        nx.draw_networkx_edges(self.G,self.pos,edgelist=self.G.edges(),width=width,alpha=0.5,edge_color='b')
        w = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_labels(self.G,self.pos,font_size=20,font_family='sans-serif')
        # nx.draw_networkx_edge_labels(self.G,self.pos,edge_labels=w,font_size=9)
        plt.axis('off')
        plt.show()
        # plt.savefig("result-iteration-%.png" %(iteration))
       

        



def main():
    # d = {'a':['b', 'c'], 'b':['a', 'c'], 'c':['a', 'b']}
    # d = {1:[2, 6, 7], 2:[1, 3, 7], 3:[2, 7, 4], 4:[3, 5, 7], 5:[4, 6, 7], 6:[1, 5, 7], 7:[1, 2, 3, 4, 5, 6]}
    # d = {1:[2, 3, 4], 2:[1, 3, 4], 3:[1, 2, 4], 4:[1, 2, 3]}
    net = Network(graph=d, weight_mode="uniform", broadcast_prob=1.0, seed=None)

    for i in range(1500):
        origin, gossiper, victim = net.init_round()
        if victim == None:  # gossip fail !
            continue
        times = 10
        for _ in range(times):
            # print (origin, victim, gossiper)
            # print (net.G[origin][victim]['weight'])
            # print (net.G[victim][origin]['weight'])
            net.update_relation(origin, victim, gossiper)
            origin = gossiper            
            gossiper = net.get_gossiper(origin, victim)
            if gossiper == None:
                break
    net.draw(i)
       


if __name__ == '__main__':
    main()
