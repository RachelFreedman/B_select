import random
import numpy as np
from decimal import Decimal

def update_belief(belief, expectation, item_to_update, item_to_compare, teacher, item_preferred, reward_vals):
    '''
        Update belief distribution over reward of specified item based on specified query and label
        Arguments:
            belief: ((N,R_max+1) ndarray) prior over reward values for each item
            expectation: ((N) array) expectation over belief for each item
            item_to_update: (Item) item to update belief distribution for
            item_to_compare: (Item) item that was used as comparison in teacher query
            teacher: (Teacher) teacher that was queried
            item_preferred: (Item) item that the teacher preferred out of (item_to_update, item_to_compare)
            reward_vals: (int array) list of possible reward values
        Return:
            belief: ((N,R_max+1) ndarray) posterior over reward values for each item
    '''

    item_index_update = int(item_to_update)
    item_index_compare = int(item_to_compare)
    alternative_selected = 0 if item_to_update == item_preferred else 1

    likelihood = [calc_likelihood(teacher.beta, r, expectation[item_index_compare], alternative_selected) for r in reward_vals]
    unnormalized_posterior = np.multiply(belief[item_index_update], likelihood)
    normalised_posterior = unnormalized_posterior / np.sum(unnormalized_posterior)

    belief[item_index_update] = normalised_posterior

    return belief


def calc_likelihood(beta, r0, r1, alternative_selected):
    val_0 = Decimal(beta * r0).exp()
    val_1 = Decimal(beta * r1).exp()

    if alternative_selected == 0:
        likelihood = float(val_0 / (val_0 + val_1))
    elif alternative_selected == 1:
        likelihood = float(val_1 / (val_0 + val_1))
    else:
        print("ERROR: invalid alternative:", alternative_selected)

    return likelihood


def update_expectation(expectation, belief, reward_vals, indices_to_update):
    if type(indices_to_update) != list:
        # make it a list
        indices_to_update = [indices_to_update]
    for i in indices_to_update:
        item_belief = belief[i]
        expectation[i] = np.dot(item_belief, reward_vals)

    return expectation

class Teacher(object):
    def __init__(self, beta, seed=None):
        self.beta = beta
        random.seed(seed)

    def get_beta(self):
        return self.beta

    def get_dist(self, r0, r1):
        ''' Return Boltzmann-rational distribution over alternatives i0 (with reward r0) and i1 (reward r1) '''
        val_0 = Decimal(self.beta * r0).exp()
        val_1 = Decimal(self.beta * r1).exp()
        prob_0 = val_0/(val_0+val_1)
        prob_1 = val_1/(val_0+val_1)
        return [prob_0, prob_1]

    def sample_dist(self, r0, r1):
        dist = self.get_dist(r0, r1)
        r = random.random()
        if r < dist[0]:
            return 0
        return 1