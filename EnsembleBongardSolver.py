'''
Author: Tanner Bohn, Joey

Date: 2018 June 21

Purpose:

- solve Bongard problems by embedding the images with an ensemble of random convolutional encoders
    and then using a logistic regression to classify left/right
'''

from sklearn.linear_model import LogisticRegression
import numpy as np
from RandomConvEncoder import *

class EnsembleBongardSolver:

    def __init__(self, problem_accessor, img_size, n_models, sub_runs, C, conv_depth, conv_nb_kernels, conv_kernel_size, dense_depth, dense_width):

        self.problem_accessor = problem_accessor

        self.confidence_threshold = 0. # no threshold -- consider every problem attempted

        self.img_size = img_size
        self.n_models = n_models
        self.sub_runs = sub_runs
        self.C = C

        self.conv_depth = conv_depth
        self.conv_nb_kernels = conv_nb_kernels
        self.conv_kernel_size = conv_kernel_size
        self.dense_depth = dense_depth
        self.dense_width = dense_width

        self.classifier = LogisticRegression(penalty='l1', C=self.C)

        self.feature_extractors = [RandomConvEncoder(img_size = self.img_size, 
                conv_depth=conv_depth, conv_nb_kernels=conv_nb_kernels, conv_kernel_size=conv_kernel_size, dense_depth=dense_depth, dense_width=dense_width) for _ in range(self.n_models)]

    def solve_problems(self, known_tiles, unknown_tiles):

        # get the image vectors associated with the bongard tiles we can work with
        left_img_vecs, right_img_vecs, new_img_vecs = self.problem_accessor.get_img_vecs(known_tiles, unknown_tiles)

        nb_new = len(new_img_vecs)

        # the class labels will be the same every time, so just create them once
        y = [0 for _ in left_img_vecs]+[1 for _ in right_img_vecs]


        # maintain list of guesses and guess confidences for each unknown tile
        guesses = [[] for _ in new_img_vecs]
        confidences = [[] for _ in new_img_vecs]


        # iterate through all of the feature extractors in the ensemble and predict solutions using each one
        for feature_extractor in self.feature_extractors:

            # get image embeddings from random convolutional encoder
            left_embeddings  = feature_extractor.extract_features(np.array(left_img_vecs))
            right_embeddings = feature_extractor.extract_features(np.array(right_img_vecs))
            new_embeddings   = feature_extractor.extract_features(np.array(new_img_vecs))

            # concatenate the embeddings for the known left and right tiles
            # along with y (as defined above), this will be the training data
            X = [emb for emb in left_embeddings] + [emb for emb in right_embeddings]

            # since the resultant classifier after training may be slightly different each time,
            # run it multiple times for a more consistent answer
            for _ in range(self.sub_runs):
                clf = self.classifier

                _ = clf.fit(X, y)

                y_probs = clf.predict_proba(new_embeddings)

                for i_tile in range(nb_new):
                    
                    y_pred = 0 if y_probs[i_tile][0] > 0.5 else 1
                    
                    confidence = max(y_probs[i_tile])

                    guesses[i_tile].append(y_pred)
                    confidences[i_tile].append(confidence)

        # combine all of the individual guesses for each unknown tile
        final_guesses = np.round(np.average(guesses, axis=-1))

        # to calculate the confidence for each of the final tile predictions, take the average confidence
        #    of each of the guesses which match the final guesses
        confidence_weights = [np.array(guesses[layer]) == final_guesses[layer] for layer in range(len(guesses))]
        confidence_weights = np.array(confidence_weights)
        final_confidences = np.average(confidences, axis=-1, weights=confidence_weights)

        # produce a list of "left"/"right"/None predictions for the unknown tiles
        sides = []

        for guess, conf in zip(final_guesses, final_confidences):

            if conf <= self.confidence_threshold:
                side = None
            elif guess == 0:
                side = "left"
            elif guess == 1:
                side = "right"

            sides.append(side)


        return sides
