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
        starters = np.random.choice(self.players, n_players)
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