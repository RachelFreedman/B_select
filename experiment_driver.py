import random as r
import time
import logging
from utils import *
from selection import *

def run(B, Q, N, R_max, items, seed=None, R_dict=None, plot=False):
    '''
        B: teacher betas
        Q: number of queries
        N: number of experiments
        R_max: maximum possible reward
        items: set of items that reward is defined for
        seed: random seed (optional)
        R_dict: reward function in range [0, R_max] (optional)
    '''

    # configure python logger to write to log.txt and csvlogger to write to eval.csv
    csvlogger = initialize_logging()
    R_set = True if not R_dict else False

    logging.info("running experiment {experimentID} with parameters: B={B}, Q={Q}, N={N}, R_max={R_max}, items={items}, seed={seed}"
                 .format(experimentID=csvlogger.experimentID, B=B, Q=Q, N=N, R_max=R_max, items=items, seed=seed))

    start_time = time.time()

    for run in range(N):

        logging.info("beginning run {}/{} with seed {}".format(run, N-1, seed+run))

        # initialize
        T_list, R_dict, reward_vals = initialize(B, R_max, seed+run, R_dict, items, R_set)

        logging.info("run {} reward function: {}".format(run, R_dict))

        # create prior
        belief, expectation = create_uniform_prior(R_max, R_dict, reward_vals)

        prior = belief.copy()

        if plot:
            plot_belief(prior, R_dict, expectation)

        # run inference
        for q in range(Q):

            logging.info("making query {}/{}".format(q, Q-1))

            # randomly sample teacher, query and label
            teacher = random.choice(T_list)
            query = random.sample(items, 2)
            label = teacher.sample_dist(R_dict[query[0]], R_dict[query[1]])

            # rename items for clarity
            first_item = query[0]
            second_item = query[1]
            item_preferred = first_item if label == 0 else second_item

            # log query
            feedback_likelihood_given_truth = calc_likelihood(teacher.beta, R_dict[first_item], R_dict[second_item], label)
            logging.info("teacher {} with beta {} gave label {} to query {}, preferring {} (true rewards: [{}, {}], label likelihood: {:.0%})"
                  .format(T_list.index(teacher), teacher.beta, label, query, item_preferred.name, R_dict[first_item],
                          R_dict[second_item], feedback_likelihood_given_truth))

            # update belief for both items, using current expectation
            prior_weight_truth = np.asarray([belief[int(item)][R_dict[item]] for item in items])
            belief = update_belief(belief, expectation, item_to_update=first_item, item_to_compare=second_item,
                                   teacher=teacher, item_preferred=item_preferred, reward_vals=reward_vals)
            belief = update_belief(belief, expectation, item_to_update=second_item, item_to_compare=first_item,
                                   teacher=teacher, item_preferred=item_preferred, reward_vals=reward_vals)

            # update expectation
            expectation = update_expectation(expectation, belief, reward_vals, [int(first_item), int(second_item)])

            # log update
            posterior_weight_truth = np.asarray([belief[int(item)][R_dict[item]] for item in items])
            chng_weight_truth = posterior_weight_truth - prior_weight_truth
            csvlogger.record_query(run=run, q=q, beta=teacher.beta, query=query, feedback_likelihood=feedback_likelihood_given_truth,
                                   truth_weights=posterior_weight_truth, truth_weights_chng=chng_weight_truth, expectations=expectation)
            logging.info("performed bayesian update; change in weight on ground truth reward for each item: {}".format(chng_weight_truth))


        # record result
        posterior = belief.copy()
        diff = posterior - prior
        total_diff = sum(sum(abs(diff)))

        print("run {}/{} complete".format(run+1, N))
        logging.info("run {} complete".format(run))
        logging.info("total probability mass moved in run {}: {:.3f}".format(run, total_diff))
        logging.info("average of ground truth weight (across {} items) in run {}: {:.3f}".format(len(items), run, np.mean(posterior_weight_truth)))

        if plot:
            plot_belief(posterior, R_dict, expectation)

    end_time = time.time()
    csvlogger.write_record_to_csv()
    print("finished {} run(s) of {} queries each in {} seconds".format(N, Q, str(round((end_time - start_time), 2))))
    logging.info("finished {} run(s) of {} queries each in {} seconds".format(N, Q, str(round((end_time - start_time), 2))))

    return csvlogger.experimentID

def initialize(B, R_max, seed, R_dict, items, R_set):
    r.seed(seed)
    np.random.seed(seed)
    set_pyplot_colors()

    T_list = []
    for beta in B:
        T_list.append(Teacher(beta, seed))

    if R_set:
        R_dict = create_reward_dict(items, R_max)

    reward_vals = np.linspace(0, R_max, R_max + 1)

    return T_list, R_dict, reward_vals

def initialize_logging():
    experimentID = time.strftime("%Y%m%d%H%M%S")

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './logs')
    directory = os.path.join(path, experimentID)
    if not (os.path.exists(directory)):
        os.makedirs(directory)
    filename = os.path.join(directory, "log.txt")

    logging.basicConfig(filename=filename,
                        format='[%(asctime)s|%(module)-1s|%(funcName)-1s|%(levelname)-1s] %(message)s',
                        datefmt='%H:%M:%S', level=logging.INFO)
    print("Logging experiment {} at {}".format(experimentID, filename))

    return CSVLogger(directory, experimentID)

def create_uniform_prior(R_max, R_dict, reward_vals):
    # create a uniform dist over [0, R_max] (inclusive) for each item
    N = len(R_dict.keys())
    uniform_val = 1 / (R_max + 1)
    belief = np.full((N, R_max + 1), uniform_val)
    expectation = [np.dot(item_belief, reward_vals) for item_belief in belief]

    return belief, expectation