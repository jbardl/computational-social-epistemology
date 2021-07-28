import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
from p5 import *

class Player:
    def __init__(self, name):
        self.name = name
        self.memory_vector = [np.random.randint(1,4)]
        self.success = 0
    
    def add_strategy(self, strategy):
        self.memory_vector.append(strategy)
        
    def erase_memory(self):
        self.memory_vector = []

        
class CoordinationGame:
    def __init__(self, player_1, player_2):
        self.player_1 = player_1
        self.player_2 = player_2
        self.players = [player_1, player_2]
        
    def play(self):
        player_1_strat = np.random.choice(self.player_1.memory_vector)
        player_2_strat = np.random.choice(self.player_2.memory_vector)
        
        if player_1_strat == player_2_strat:
            for player in self.players:
                player.erase_memory()
                player.add_strategy(player_1_strat)
                player.success = 1
        else:
            if player_2_strat not in self.player_1.memory_vector:
                self.player_1.add_strategy(player_2_strat)
            else:
                pass
            if player_1_strat not in self.player_2.memory_vector:
                self.player_2.add_strategy(player_1_strat)
            else:
                pass
            self.player_1.success = 0
            self.player_2.success = 0

class Simulation:
    def __init__(self, players, network, rounds):
        self.network = network
        nx.set_node_attributes(self.network, players, 'jugador')
        self.players = [self.network.nodes[node]['jugador'] for node in self.network.nodes]
        self.rounds = rounds
        self.successes = []
        self.last_round = 0
    
    def update_successes(self):
        self.successes.append(sum([player.success for player in self.players]))
        
    def setup(self):
        self.successes = []
        self.last_round = 0
        for player in self.players:
            player.erase_memory()
            player.memory_vector = [np.random.randint(1,4)]
        
    def include_social_movement(self, n_players):
        starters = np.random.choice(self.players, n_players, replace=False)
        for starter in starters:
            starter.erase_memory()
            starter.memory_vector = [5]
    
    def cycle(self):
        for node in self.network:
            player = self.network.nodes[node]['jugador']
            other_player = self.network.nodes[np.random.choice(list(self.network.neighbors(node)))]['jugador']
            game = CoordinationGame(player, other_player)
            game.play()
        self.update_successes()
        
    def go(self, v=True):
        iterator = range(self.rounds)
        for i in (tqdm(iterator) if v else iterator):
            self.cycle()
            
            last_round = i
            
            if self.successes == len(self.players):
                break
        
    def _reached_stability(self):
        player_strats = [player.memory_vector for player in self.players]
        if player_strats.count(player_strats[0])  == len(player_strats):
            return 1
        else:
            return 0
            
    def plot_successes(self):
        data = go.Scatter(
            x=list(range(1,self.rounds+1)),
            y=self.successes
        )

        layout = go.Layout(
            title='Éxitos en la coordinación por ciclos de la simulación',
            xaxis_title='Ticks',
            yaxis_title='Éxitos',
            template='plotly_dark'
        )

        fig = go.Figure(data=data, layout=layout)
        py.offline.iplot(fig)
        
    def plot_network(self):
        kwargs = {'width':0.2, 'node_size':50, 'with_labels':False, "node_color":"red"}#
        plt.figure(figsize=(10,7))
        nx.draw_kamada_kawai(self.network, **kwargs)
        plt.show()
        

def setup_watts_strogatz(n, k, p, rounds):
    players = {num:Player(num) for num in list(range(n))}
    network = nx.watts_strogatz_graph(n, k, p)
    return Simulation(players, network, rounds)

def setup_random_network(n, p, rounds):
    players = {num:Player(num) for num in list(range(n))}
    network = nx.erdos_renyi_graph(n, p)
    return Simulation(players, network, rounds)
        
        
def test_sim(n_tests, sim):
    sim_stabilized = []
    for n_test in tqdm(range(n_tests)):
        sim.setup()
        sim.go(v=False)
        sim_stabilized.append(sim._reached_stability())
    return sim_stabilized

def get_node_color(player):
    if len(player.memory_vector) == 1:
        if player.memory_vector == [1]:
            return 'red'
        elif player.memory_vector == [2]:
            return 'blue'
        elif player.memory_vector == [5]:
            return 'brown'
        else:
            return 'yellow'
    elif len(player.memory_vector) == 2:
        return 'magenta'
    else:
        return 'green'

sim = setup_watts_strogatz(100, 4, 0, 500)
graph = sim.network
layout_pos = nx.spring_layout(graph)
layout = {node:(layout_pos[node]*300)+420 for node in layout_pos}
sim.include_social_movement(90)

def setup():
    size(898, 899)
    background(24)        
    
def draw():
    fill(161, 181, 108)
    # text_size(10)
    sim.cycle()
    # text(str(sim.last_round), (0,10))
    for node in graph:
        color = get_node_color(sim.network.nodes[node]['jugador'])
        fill(color)
        circle(layout[node][0], layout[node][1], 20)
        for neighbor in nx.neighbors(sim.network, node):
            stroke(51)
            line(layout[node][0], layout[node][1], layout[neighbor][0] ,layout[neighbor][1])
    

    
if __name__ == '__main__':
    run()