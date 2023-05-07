import pulp
import sys
import numpy as np

class MDP:
    def __init__(self, mdp_file: str) -> None:
        with open(f'MDPs/{mdp_file}', 'r') as file:
            for line in file:
                vals = line.split()
                if vals[0] == 'states':
                    self.num_states = int(vals[1])
                    # self.values = np.zeros(self.num_states) # State value function
                    # self.policy = np.zeros(self.num_states, dtype=int) # Our policy
                elif vals[0] == 'actions':
                    self.num_actions = int(vals[1])
                    self.transition_probs = [[[{} for _ in range(self.num_states)] for _ in range(self.num_actions)] for _ in range(self.num_states)]
                    '''self.transition_probs[s][a][s'][r] = p(s', r | s, a)'''
                elif vals[0] == 'tran':
                    self.transition_probs[int(vals[1])][int(vals[2])][int(vals[3])][np.float64(vals[4])] = np.float64(vals[5])
                elif vals[0] == 'gamma':
                    self.gamma = np.float64(vals[1])
                else:
                    raise Exception('File Format is incorrect')
    
    def solve(self) -> None:
        '''
        We solve the Bellman Optimality Equations explicitly
        We have v(s) = \max_{a \in A(s)} \sum_{r, s' \in S} p(s', r | s, a) (r + \gamma v(s')), for all s \in S
        We need to convert this to a Linear Programming Problem. v(s), for all s \in S are our variables
        Firstly, v(s) >= \sum_{r, s' \in S} p(s', r | s, a) (r + \gamma v(s')), for all a \in A(s), for all s \in S
        To make this system of inequalities equivalent to the BOEs, we need that for all s \in S, there exists a \in A(s) st v(s) = \sum_{r, s' \in S} p(s', r | s, a) (r + \gamma v(s'))
        '''

# Enter file name as command line argument
mdp = MDP(sys.argv[1])
mdp.solve()
mdp.print_solution(f'sol-{sys.argv[1]}')