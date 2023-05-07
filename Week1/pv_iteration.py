# AUTHOR: ASHWIN ABRAHAM

import sys
import numpy as np

class MDP:
    NUM_POLICY_EVAL_PER_IMP = 100 if len(sys.argv) <= 2 else int(sys.argv[2]) # Set this to 1 for value iteration
    NUM_POLICY_IMP = 100 if len(sys.argv) <= 3 else int(sys.argv[3])

    def __init__(self, mdp_file: str) -> None:
        with open(f'MDPs/{mdp_file}', 'r') as file:
            for line in file:
                vals = line.split()
                if vals[0] == 'states':
                    self.num_states = int(vals[1])
                    self.values = np.zeros(self.num_states) # State value function
                    self.policy = np.zeros(self.num_states, dtype=int) # Our policy
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
        for _ in range(MDP.NUM_POLICY_IMP):
            for s in range(self.num_states):
                '''Policy Improvement'''
                self.policy[s] = np.argmax([np.sum([np.sum([self.transition_probs[s][a][s1][r]*(r + self.gamma * self.values[s1]) for r in self.transition_probs[s][a][s1]]) for s1 in range(self.num_states)]) for a in range(self.num_actions)]) # type: ignore
            
            for _ in range(MDP.NUM_POLICY_EVAL_PER_IMP):
                for s in range(self.num_states):
                    '''Policy Evaluation'''
                    self.values[s] = np.sum([np.sum([self.transition_probs[s][self.policy[s]][s1][r]*(r + self.gamma * self.values[s1]) for r in self.transition_probs[s][self.policy[s]][s1]]) for s1 in range(self.num_states)]) # type: ignore
    
    def print_solution(self, out_file: str):
        with open(out_file, 'w') as of:
            for s in range(self.num_states):
                of.write(f'{self.values[s]} {self.policy[s]}\n')

# Enter file name as command line argument
mdp = MDP(sys.argv[1])
mdp.solve()
mdp.print_solution(f'sol-{sys.argv[1]}')