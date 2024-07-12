import random
import itertools

def player(prev_play: str) -> str:
    """Plays rock paper scissors, called by main.py

    Args:
        prev_play (str): A string representing the opponent's last play

    Returns:
        str: Move to play
    """
    return markov(prev_play)

class MarkovChain:
    """Uses a Markov Chain to predict a system's state"""
    
    def __init__(self, length: int, state_possibilities: list[str], result_possibilities: list[str], decay=0.):
        """Initialized a Markov Chain

        Args:
            length (int): Number of previous states to combine for predictions
            state_possibilities (list[str]): A list of possible states
            result_possibilities (list[str]): A list of all possible results
            decay (float, optional): How much to lerp transition probabilities to uniform per update. Defaults to 0..
        """        
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

        
    def update(self, state: str, result: str) -> (int | None):
        """Update the transition probabilities with new data

        Args:
            state (str): State of the system
            result (str): The result following the given state

        Returns:
            Optional[int]: -1 if there is not enough data to form a combined state of required length
        """        
        self.states.append(state)
        
        if len(self.states) < self.length:
            return -1
        
        state_chain = ''.join(self.states[-self.length:]) # combine states
        probs = self.transition_matrix[state_chain] # shorthand
        
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
        
    def _probs_without_obs(self, probs: dict[str, float]) -> dict[str, float]:
        """Removes 'num_obs' dict entry from dict of transition probabilities"""
        return {i:probs[i] for i in probs if i != 'num_obs'}
    
    def get_probs(self, state_chain: str) -> dict[str, float]:
        """Returns probabilities for a given combined state"""
        return self._probs_without_obs(self.transition_matrix[state_chain])
    
    def predict_max(self, state_chain: str) -> tuple[str, float]:
        """Predicts most likely state

        Args:
            state_chain (str): Combined previous states

        Returns:
            str: prediction,
            float: estimated probability
        """
        probs = self._probs_without_obs(self.transition_matrix[state_chain])
        prediction = max(probs, key=probs.get) # type: ignore
        return prediction, probs[prediction]
    
    def predict_chance(self, state_chain: str, spice=0.) -> tuple[str, float]:
        """Selects state from possible resulting states based on probability

        Args:
            state_chain (str): Combined previous states
            spice (float): Amount to lerp probabilities to their average

        Returns:
            str: prediction,
            float: estimated probability
        """
        probs = self._probs_without_obs(self.transition_matrix[state_chain])
        weights = list(probs.values())
        
        avg = sum(weights) / len(weights)
        for i in range(len(weights)):
            weights[i] += (avg - weights[i])*spice
        
        prediction = random.choices(list(probs.keys()), weights=weights)[0]
        return prediction, probs[prediction]
    
    def chain_from_prev_state(self, prev_state: str) -> str:
        """Combines previous state with previous states to create combined state"""
        return ''.join(self.states[-self.length:][1:])+prev_state
    
    def get_next_probs(self, prev_state: str) -> dict[str, float]:
        state_chain = self.chain_from_prev_state(prev_state)
        return self.get_probs(state_chain)
    
    def predict_max_next(self, prev_state: str) -> tuple[str, float]:
        """Predicts most likely state

        Args:
            prev_state (str): Previous state

        Returns:
            str: prediction,
            float: estimated probability
        """
        state_chain = self.chain_from_prev_state(prev_state)
        return self.predict_max(state_chain)
    
    def predict_chance_next(self, prev_state: str, spice=0.) -> tuple[str, float]:
        """Selects state from possible resulting states based on probability

        Args:
            prev_state (str): Previous state
            spice (float): Amount to lerp probabilities to their average

        Returns:
            str: prediction,
            float: estimated probability
        """
        state_chain = self.chain_from_prev_state(prev_state)
        return self.predict_chance(state_chain, spice=spice)


IDEAL_RESPONSE = {'P': 'S', 'R': 'P', 'S': 'R'} # maps opponent play to ideal player play
SPICE = -2.0 # UNUSED. amount to lerp transition probabilities to their average
SPICE_BETWEEN_CHAINS = -4.0 # UNUSED. amount to lerp probabilities from different markov chain predictions to their average
CHAIN_LENGTHS = [1, 2, 3]
STATE_POSSIBILITIES = ['RR', 'RP', 'RS',
                       'PR', 'PP', 'PS',
                       'SR', 'SP', 'SS'] # all possible pairs of plays
PLAY_OPTIONS = ['R', 'P', 'S'] # possible plays
DECAY = 0.0 # amount to regress probabilities to uniform per update

def markov(prev_opp_play: str, chains: list[MarkovChain] = [], prev_play: list[str]=[''], prev_state: list[str]=['']) -> str:
    """Plays the game using markov chain predictions

    Args:
        prev_opp_play (str): Opponent's previous move
        chains (list[MarkovChain], DO NOT SET): Stores Markov Chains between moves.
        prev_play (list, DO NOT SET): Single element list to store player's previous play.
        prev_state (list, DO NOT SET): Single element list to store game's previous state.

    Returns:
        str: Move to play
    """    
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
    
    # update markov chains and make predictions
    predictions = []
    probs = []
    for chain in chains:
        code = chain.update(prev_state[0], prev_opp_play)
        
        if code == -1: # make sure markov chain has enough data
            predictions.append(random.choice(PLAY_OPTIONS))
            probs.append(1.0/len(PLAY_OPTIONS))
            continue
        
        prediction, prob = chain.predict_max_next(prev_opp_play+prev_play[0])
        predictions.append(prediction)
        probs.append(prob)
        
    # get most likely outcome from predictions
    max_prob_index = probs.index(max(probs))
    prediction = predictions[max_prob_index]
    
    guess = IDEAL_RESPONSE[prediction]
    
    # update prev vars
    prev_state[0] = prev_opp_play + prev_play[0]
    prev_play[0] = guess
    
    return guess
