import random
import os
import csv
from enum import IntEnum
from selection import Teacher
import matplotlib.pyplot as plt
import numpy as np

class Item(IntEnum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5
    G = 6
    H = 7
    I = 8
    J = 9

def set_pyplot_colors():
    color_list = []
    color_list.append('#EDBC3C')
    color_list.append('#387c4a')
    color_list.append('#07293F')
    color_list.append('#2a9a9b')
    color_list.append('#DB6222')
    color_list.append('#B43B24')

    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

def create_reward_dict(items, R_max):
    reward_dict = {}
    for item in items:
        reward_dict[item] = random.randrange(R_max)
    return reward_dict

def create_teacher_list(betas, seed=None):
    teachers = []
    for beta in betas:
        teachers.append(Teacher(beta, seed))
    return teachers

def plot_belief(belief, R_dict, expectation=None):
    for i in range(len(belief)):
        item = Item(i)
        item_belief = belief[i]
        item_expectation = expectation[i]
        item_truth = R_dict[item]

        plt.plot(item_belief, label='belief')
        if expectation:
            plt.axvline(x=item_expectation, color='r', label='expectation')
        plt.axvline(x=item_truth, color='g', label='true reward')

        plt.ylim([0, 0.1])
        plt.title('Belief over R({})'.format(item.name))
        plt.legend()
        plt.show()

def import_data(experimentID, N):
    file = "./logs/{expID}/eval.csv".format(expID=experimentID)
    print("reading {}".format(file))

    filetext = np.genfromtxt(file, delimiter=',', dtype=None, encoding='UTF-8')
    headers = filetext[0]
    run_col = np.asarray(filetext[:, 1])
    data = {}

    for run in range(N):
        data[run] = {}
        run_data = filetext[np.where(run_col == str(run))]
        for i in range(len(headers)):
            data[run][headers[i]] = run_data[:, i]

    return data


def plot_average_weight_on_ground_truth(data, B):
    query_num = data[0]['query_num'].astype(np.int)

    for run in data.keys():
        truth_weight = [np.fromstring(s[1:-1], dtype=float, sep=' ') for s in data[run]['truth_weights']]
        avg_truth_weight = np.mean(truth_weight, axis=1)
        plt.plot(query_num, avg_truth_weight, label='run {}'.format(run))

    plt.title("average belief weight on ground truth with B={}".format(B[0]))
    plt.xlabel('query')
    plt.ylabel('weight')
    plt.legend()


def plot_belief_expectation(data, B, items, R_dict=None):
    query_num = data[0]['query_num'].astype(np.int)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for run in data.keys():
        for item in items:
            expectations = np.asarray(
                [np.fromstring(s[1:-1], dtype=float, sep=' ') for s in data[run]['expectations']])

            color = colors[int(item) % len(colors)]
            plt.plot(query_num, expectations[:, int(item)], color=color, linestyle='-',
                     label='item {} estimate'.format(item.name))
            if R_dict:
                plt.plot(query_num, np.full((len(expectations), 1), R_dict[item]), color=color, linestyle=':',
                     label='item {} truth'.format(item.name))

        plt.title("expectation of belief distributions, B={}, run={}".format(B[0], run))
        plt.xlabel('query')
        plt.ylabel('reward value')
        plt.legend()
        plt.show()


def plot_average_change_in_weight_on_ground_truth(data, B):
    query_num = data[0]['query_num'].astype(np.int)

    for run in data.keys():
        truth_weights_chng = [np.fromstring(s[1:-1], dtype=float, sep=' ') for s in data[run]['truth_weights_chng']]
        avg_truth_weight = np.mean(truth_weights_chng, axis=1)
        plt.plot(query_num, avg_truth_weight, label='run {}'.format(run), alpha=0.5)

    plt.plot(query_num, np.full((avg_truth_weight.shape), 0), color='black')

    plt.title("average belief chng on ground truth with B={}".format(B[0]))
    plt.xlabel('query')
    plt.ylabel('weight')
    plt.legend()

class CSVLogger():

    def __init__(self, directory, experimentID):
        self.experimentID = experimentID
        self.eval_data = []

        filename = "eval.csv"
        self.path = directory + os.path.sep + filename
        print("Recording eval for experiment {} at {}".format(self.experimentID, self.path))

    def record_query(self, run, q, beta, query, feedback_likelihood, truth_weights, truth_weights_chng, expectations):
        record = {
            "experimentID": self.experimentID,
            "run": run,
            "query_num": q,
            "teacher_beta": beta,
            "query": query[0].name + "; " + query[1].name,
            "feedback_likelihood": feedback_likelihood,
            "truth_weights": truth_weights,
            "truth_weights_chng": self.convert_truth_weights_chng_to_string_manually_please(truth_weights_chng),
            "expectations": np.asarray(expectations).copy()
        }
        self.eval_data.append(record.copy())

    # convert truth_weights_chng array to string in normal np.tostring format with NO NEWLINES
    def convert_truth_weights_chng_to_string_manually_please(self, truth_weights_chng):
        s = "["  + str(truth_weights_chng[0])
        if len(truth_weights_chng) > 1:
            for e in truth_weights_chng[1:]:
                s += " " + str(e)
        s += "]"
        return s


    def write_record_to_csv(self):
        csv_columns = ["experimentID", "run", "query_num", "teacher_beta", "query", "feedback_likelihood",
                       "truth_weights", "truth_weights_chng", "expectations"]

        with open(self.path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns, lineterminator='\n')
            writer.writeheader()
            for data in self.eval_data:
                writer.writerow(data)

if __name__ == '__main__':
    items = [Item.D, Item.E, Item.F]
