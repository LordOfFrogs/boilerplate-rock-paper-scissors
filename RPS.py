import random
import itertools

def player(prev_play):
    return markov(prev_play)

class MarkovChain:
    def __init__(self, length, state_possibilities, result_possibilities, decay=0.):
        self.length = length
        self.decay = decay
        self.states = []
        self.transition_matrix = {} # for each state chain, associate a probability with each result
        self.poss_state_chains = list(map(''.join, itertools.product(state_possibilities, repeat=length)))

        for state_chain in self.poss_state_chains:
            self.transition_matrix[state_chain] = {}
            for poss in result_possibilities:
                self.transition_matrix[state_chain][poss] = 1./len(result_possibilities)
            self.transition_matrix[state_chain]['num_obs'] = 0

        
    def update(self, state, result):
        self.states.append(state)

        if len(self.states) < self.length:
            return -1
        state_chain = ''.join(self.states[-self.length:])
        probs = self.transition_matrix[state_chain]
        
        # update transition matrix probabilities
        for key in list(probs.keys())[:-1]: # exclude num_obs
            probs[key] *= probs['num_obs'] # prob -> count
            
            # update for result
            if key == result:
                probs[key] += 1
                
            probs[key] /= probs['num_obs'] + 1 # count -> prob
        
        probs['num_obs'] += 1
        
        # decay to equal probabilities
        for probs in self.transition_matrix.values():
            for key in probs.keys():
                num_probs = len(probs.keys()) - 1
                if key != 'num_obs':
                    probs[key] += (1.0/num_probs - probs[key])*self.decay
        
    def _probs_without_obs(self, probs):
        return {i:probs[i] for i in probs if i != 'num_obs'}
    
    def get_probs(self, state_chain):
        return self._probs_without_obs(self.transition_matrix[state_chain])
    
    def predict_max(self, state_chain):
        probs = self._probs_without_obs(self.transition_matrix[state_chain])
        prediction = max(probs, key=probs.get)
        return prediction, probs[prediction]
    
    def predict_chance(self, state_chain, spice=0.):
        probs = self._probs_without_obs(self.transition_matrix[state_chain])
        weights = list(probs.values())
        
        avg = sum(weights) / len(weights)
        for i in range(len(weights)):
            weights[i] += (avg - weights[i])*spice
        
        prediction = random.choices(list(probs.keys()), weights=weights)[0]
        return prediction, probs[prediction]
    
    def chain_from_prev_state(self, prev_state):
        return ''.join(self.states[-self.length:][1:])+prev_state
    
    def get_next_probs(self, prev_state):
        state_chain = self.chain_from_prev_state(prev_state)
        return self.get_probs(state_chain)
    
    def predict_max_next(self, prev_state):
        state_chain = self.chain_from_prev_state(prev_state)
        return self.predict_max(state_chain)
    
    def predict_chance_next(self, prev_state, spice=0.):
        state_chain = self.chain_from_prev_state(prev_state)
        return self.predict_chance(state_chain, spice=spice)


IDEAL_RESPONSE = {'P': 'S', 'R': 'P', 'S': 'R'}
SPICE = -2.0
SPICE_BETWEEN_CHAINS = -4.0
CHAIN_LENGTHS = [1, 2, 3]
STATE_POSSIBILITIES = ['RR', 'RP', 'RS',
                       'PR', 'PP', 'PS',
                       'SR', 'SP', 'SS']
PLAY_OPTIONS = ['R', 'P', 'S']
DECAY = 0.0

def markov(prev_opp_play: str, chains: list[MarkovChain] = [], prev_play=[''], prev_state=[''], prev_player=[0]):
    # first moves
    if prev_opp_play == '':
        chains.clear()
        prev_state[0] = ''
        for length in CHAIN_LENGTHS:
            chains.append(MarkovChain(length, STATE_POSSIBILITIES, PLAY_OPTIONS, decay=DECAY))
        guess = random.choice(['R', 'P', 'S'])
        prev_play[0] = guess
        return guess

    if prev_state[0] == '':
        prev_state[0] = prev_opp_play + prev_play[0]
        guess = random.choice(['R', 'P', 'S'])
        prev_play[0] = guess
        return guess
        
    predictions = []
    probs = []
    for chain in chains:
        code = chain.update(prev_state[0], prev_opp_play)
        if code and code < 0: # make sure markov chain has enough data
            predictions.append(random.choice(PLAY_OPTIONS))
            probs.append(1.0/len(PLAY_OPTIONS))
            continue
        prediction, prob = chain.predict_max_next(prev_opp_play+prev_play[0])
        predictions.append(prediction)
        probs.append(prob)
        
    prev_state[0] = prev_opp_play + prev_play[0]
    
    max_prob_index = probs.index(max(probs))
    prediction = predictions[max_prob_index]
    
    guess = IDEAL_RESPONSE[prediction]
    
    # update prev vars
    prev_play[0] = guess
    
    return guess

MAX_CHAIN_LEN = 250

def markov_manual(prev_play, self_history=[], opponent_history=[]):
    # first moves
    if prev_play == '':
        self_history.clear()
        opponent_history.clear()
        guess = random.choice(['R', 'P', 'S'])
        self_history.append(guess)
        return guess
    
    if len(opponent_history) <= 2:
        opponent_history.append(prev_play)
        guess = random.choice(['R', 'P', 'S'])
        self_history.append(guess)
        return guess
    opponent_history.append(prev_play)
    #print(prev_play)
    
    # record play state history
    prev_states = []
    self_totals = {'R': 0, 'P': 0, 'S': 0}
    for i in range(len(self_history)-2):
        self_totals[self_history[i]] += 1
        most_frequent = max(self_totals, key=self_totals.get)
        prev_states.append(opponent_history[i-1]+self_history[i-1] + opponent_history[i]+self_history[i] + most_frequent)
    
    if len(prev_states) > MAX_CHAIN_LEN:
        prev_states = prev_states[-MAX_CHAIN_LEN:]
    
    # initialize probabilities
    probs = {}
    for state in prev_states:
        probs[state] = {'R': 0, 'P': 0, 'S': 0}
    
    # total up responses
    for i in range(len(prev_states)):
        response = opponent_history[i+1]
        probs[prev_states[i]][response] += 1
        
    #print(self_history, opponent_history, probs)
    
    # normalize to probabilities
    for state, response_probs in probs.items():
        total = response_probs['R'] + response_probs['P'] + response_probs['S']
        response_probs['R'] /= total
        response_probs['P'] /= total
        response_probs['S'] /= total
    
    prev_state = prev_states[-1]
    predict_probs = list(probs[prev_state].values())
    # add spice
    avg = (predict_probs[0]+predict_probs[1]+predict_probs[2])/3
    predict_probs[0] += (avg-predict_probs[0])*SPICE
    predict_probs[1] += (avg-predict_probs[1])*SPICE
    predict_probs[2] += (avg-predict_probs[2])*SPICE
    
    prediction = random.choices(list(probs[prev_state].keys()), weights=predict_probs)[0]
    #print(probs[prev_state], end=' - ')
    guess = IDEAL_RESPONSE[prediction]
    self_history.append(guess)
    if len(self_history) == 999:
        for key, value in probs.items():
            value['R'] = round(value['R'], 2)
            value['P'] = round(value['P'], 2)
            value['S'] = round(value['S'], 2)
        #print('\n', probs)
    return guess