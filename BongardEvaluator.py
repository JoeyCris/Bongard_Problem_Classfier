'''
Author: Joey, Tanner Bohn

Date: 2018 June 19 (updated June 21)

This evaluation class offers 3 types of evaluation for bongard solvers:
    1. "dozen split":
        - determine how well the solver performs at finding a way to split the 12 original tiles
        - allow the solver to both train and test on all 12 tiles

    2. "classify manual":
        - determine how well the solver is able to classify new tiles into the correct side
        - allow the solver to train on the 12 original tiles, and then test on 2 new ones

    3. "remove and reclassify":
        - tries to approximate how well the solver is able to generalize for a bongard problem, but
            without needing manually drawn tiles
        - allow solver to train on 5 left and five right, and then test of the 2 that were taken out
            - this is done for all 36 possible pairs of tiles
        - takes much longer than the previous two methods

To utilize this class, a bongard solver needed a .solve_problems method, which takes the following parameters:
    - 'known_tiles': this is a list specifying which tiles the solver can observe before trying to
        classify unknown tiles. See code for details
    - 'unknown_tiles': this list has the same format as 'known_tiles', and specifies which tiles the
        bongard solver need to try and classify. See code for details

The .solve_problems method of a bongard solver should return a list of "left"/"right"/None entries -- corresponding to each tile in the unknown_tiles list.

The classification of None can be used when the solver does not attempt to classify that tile (due to a low confidence or other reason).

'''


import numpy as np
import itertools

class BongardEvaluator:

    def __init__(self):

        return


    def dozen_split(self, solver, problem_numbers, verbose=False):

        # for each of the bongard problems specified, store a score
        problem_scores = []


        if verbose:
            print("problem\tfrac_attempted\tfrac_correct")

        for p_num in problem_numbers:

            # generate lists of known and unknown tiles
            known_tiles = []
            unknown_tiles = []
            for field, index in itertools.product(["real_left", "real_right"], list(range(6))):
                known_tiles.append((p_num, field, index))
                unknown_tiles.append((p_num, field, index))


            results = self._evaluate_problem(solver, known_tiles, unknown_tiles)

            problem_scores.append(int(results['frac_correct'] == 1))

            if verbose:
                print("{}\t{:.4f}\t{:.4f}".format(p_num, results['frac_attemped'], results['frac_correct']))

        avg_score = np.average(problem_scores)

        return avg_score

    def all_split(self, solver, problem_numbers, verbose=False):

        # for each of the bongard problems specified, store a score
        problem_scores = []

        # these setting are constant for all problems, just need to change the problem number
        #known_tiles = {"real_left":[0, 1, 2, 3, 4, 5], "real_right":[0, 1, 2, 3, 4, 5], "manual_left":[0], "manual_right":[0]}
        #unknown_tiles = {"real_left":[0, 1, 2, 3, 4, 5], "real_right":[0, 1, 2, 3, 4, 5], "manual_left":[0], "manual_right":[0]}

        if verbose:
            print("problem\tfrac_attempted\tfrac_correct")

        for p_num in problem_numbers:

            # generate lists of known and unknown tiles
            known_tiles = []
            unknown_tiles = []
            for field, index in itertools.product(["real_left", "real_right"], list(range(6))):
                known_tiles.append((p_num, field, index))
                unknown_tiles.append((p_num, field, index))

            known_tiles.append((p_num, "manual_left", 0))
            known_tiles.append((p_num, "manual_right", 0))
            unknown_tiles.append((p_num, "manual_left", 0))
            unknown_tiles.append((p_num, "manual_right", 0))

            results = self._evaluate_problem(solver, known_tiles, unknown_tiles)

            problem_scores.append(int(results['frac_correct'] == 1))

            if verbose:
                print("{}\t{:.4f}\t{:.4f}".format(p_num, results['frac_attemped'], results['frac_correct']))

        avg_score = np.average(problem_scores)

        return avg_score


    def classify_manual(self, solver, problem_numbers, verbose=False):

        # for each of the bongard problems specified, store a score
        problem_scores = []

        # these setting are constant for all problems, just need to change the problem number
        #known_tiles = {"real_left":[0, 1, 2, 3, 4, 5], "real_right":[0, 1, 2, 3, 4, 5], "manual_left":[], "manual_right":[]}
        #unknown_tiles = {"real_left":[], "real_right":[], "manual_left":[0], "manual_right":[0]}

        if verbose:
            print("problem\tfrac_attempted\tfrac_correct")

        for p_num in problem_numbers:

            # generate lists of known and unknown tiles
            known_tiles = []
            unknown_tiles = []
            for field, index in itertools.product(["real_left", "real_right"], list(range(6))):
                known_tiles.append((p_num, field, index))

            unknown_tiles.append((p_num, "manual_left", 0))
            unknown_tiles.append((p_num, "manual_right", 0))

            results = self._evaluate_problem(solver, known_tiles, unknown_tiles)

            problem_scores.append(results['frac_correct'])

            if verbose:
                print("{}\t{:.4f}\t{:.4f}".format(p_num, results['frac_attemped'], results['frac_correct']))

        avg_score = np.average(problem_scores)

        return avg_score

    def remove_and_reclassify(self, solver, problem_numbers, verbose=False, sample_frac=1.):

        # for each of the bongard problems specified, store a score
        problem_scores = []

        if verbose:
            print("problem\tfrac_correct")

        for p_num in problem_numbers:

            sub_scores = []

            all_left_tiles = [(p_num, "real_left", index) for index in range(6)]
            all_right_tiles = [(p_num, "real_right", index) for index in range(6)]

            nb_splits = len(all_left_tiles) * len(all_right_tiles)
            used_splits = list(range(nb_splits))
            random.shuffle(used_splits)
            used_splits = set(used_splits[:int(sample_frac*nb_splits)])


            # for every variation of the problem, need different set of known and unknown tiles
            pos = 0
            for l_remove_index, r_remove_index in itertools.product(range(len(all_left_tiles)), range(len(all_right_tiles))):
                if pos not in used_splits: 
                    pos += 1
                    continue
                pos += 1

                known_tiles = []
                known_tiles.extend([tile for i, tile in enumerate(all_left_tiles) if i != l_remove_index])
                known_tiles.extend([tile for i, tile in enumerate(all_right_tiles) if i != r_remove_index])
                unknown_tiles = [all_left_tiles[l_remove_index], all_right_tiles[r_remove_index]]

                results = self._evaluate_problem(solver, known_tiles, unknown_tiles)

                sub_scores.append(results['frac_correct'])


            problem_score = np.average(sub_scores)

            problem_score_stddev = np.std(sub_scores)

            problem_scores.append(problem_score)

            if verbose:
                print("{}\t{:.4f}\t{:.4f}".format(p_num, problem_score_stddev, problem_score))

        avg_score = np.average(problem_scores)

        return avg_score

    def all_remove_and_reclassify(self, solver, problem_numbers, verbose=False, sample_frac=1.):

        # for each of the bongard problems specified, store a score
        problem_scores = []

        if verbose:
            print("problem\tfrac_correct")

        for p_num in problem_numbers:

            sub_scores = []

            all_left_tiles = [(p_num, "real_left", index) for index in range(6)]
            all_left_tiles.append((p_num, "manual_left", 0))
            all_right_tiles = [(p_num, "real_right", index) for index in range(6)]
            all_right_tiles.append((p_num, "manual_right", 0))


            nb_splits = len(all_left_tiles) * len(all_right_tiles)
            used_splits = list(range(nb_splits))
            random.shuffle(used_splits)
            used_splits = set(used_splits[:int(sample_frac*nb_splits)])


            # for every variation of the problem, need different set of known and unknown tiles
            pos = 0
            for l_remove_index, r_remove_index in itertools.product(range(len(all_left_tiles)), range(len(all_right_tiles))):
                if pos not in used_splits: 
                    pos += 1
                    continue
                pos += 1

                known_tiles = []
                known_tiles.extend([tile for i, tile in enumerate(all_left_tiles) if i != l_remove_index])
                known_tiles.extend([tile for i, tile in enumerate(all_right_tiles) if i != r_remove_index])
                unknown_tiles = [all_left_tiles[l_remove_index], all_right_tiles[r_remove_index]]

                results = self._evaluate_problem(solver, known_tiles, unknown_tiles)

                sub_scores.append(results['frac_correct'])


            problem_score = np.average(sub_scores)

            problem_score_stddev = np.std(sub_scores)

            problem_scores.append(problem_score)

            if verbose:
                print("{}\t{:.4f}\t{:.4f}".format(p_num, problem_score_stddev, problem_score))

        avg_score = np.average(problem_scores)

        return avg_score


    def _evaluate_problem(self, solver, known_tiles, unknown_tiles):

        # count total number of tiles to be classified
        num_to_classify = len(unknown_tiles)

        # call the bongard solver to get predictions -- first convert known/unknown tile dicts into list version
        predicted_sides = solver.solve_problems(known_tiles=known_tiles, 
                                                unknown_tiles=unknown_tiles)

        num_correct = 0
        num_attempted = 0
        for pred_side, tile in zip(predicted_sides, unknown_tiles):
            # tile = (problem num, field, index), where field is "real_left", "real_right", "manual_left", "manual_right"
            # pred_side is one of "left", "right", or None
            if pred_side in tile[1]:
                num_correct += 1

            if pred_side != None:
                num_attempted += 1

        results = {
            "frac_correct": 0. if num_attempted == 0 else 1. * num_correct / num_attempted,
            "frac_attemped": 1.*num_attempted / num_to_classify
        }

        return results


def problem_dict_to_list(problem_number, problem_dict):


    problem_list = []

    for key in sorted(problem_dict):

        for index in problem_dict[key]:
            item = (problem_number, key, index)
            problem_list.append(item)

    return problem_list

def fill_dictionary(filled_dict, filled_list):

        new_dict = dict()

        pos = 0
        for key in sorted(filled_dict):
            new_dict[key] = []

            for i in filled_dict[key]:
                new_dict[key].append(filled_list[pos])
                pos += 1

        return new_dict
