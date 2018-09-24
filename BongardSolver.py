import numpy as np
from sklearn.linear_model import LogisticRegression
import time

class BongardSolver:

	def __init__(self, problem_accessor, feature_extractor, classifier):

		self.problem_accessor = problem_accessor
		self.feature_extractor = feature_extractor
		self.classifier = classifier

		#self.penalty = penalty
		#self.C = C

	def solve_problems(self, known_tiles, unknown_tiles, return_int=False):

		left_tiles, right_tiles, new_tiles = self.problem_accessor.get_tile_labels(known_tiles, unknown_tiles)
		left_img_vecs, right_img_vecs, new_img_vecs = self.problem_accessor.get_img_vecs(known_tiles, unknown_tiles)

		nb_new = len(new_tiles)

		# the class labels will be the same every time, so just create them once
		y = [0 for _ in left_tiles]+[1 for _ in right_tiles]


		# maintain list of guesses for each unknown tile
		guesses = [[] for _ in new_tiles]


		t0 = time.time()
		#left_embeddings = self.feature_extractor.extract_features(np.array(left_img_vecs))
		#right_embeddings = self.feature_extractor.extract_features(np.array(right_img_vecs))
		#new_embeddings = self.feature_extractor.extract_features(np.array(new_img_vecs))
		left_embeddings = self.feature_extractor.extract_features_by_label(left_tiles)
		right_embeddings = self.feature_extractor.extract_features_by_label(right_tiles)
		new_embeddings = self.feature_extractor.extract_features_by_label(new_tiles)
		t1 = time.time()

		#print("\tSOLVE 0:", t1 - t0)

		X = [emb for emb in left_embeddings] + [emb for emb in right_embeddings]

	
		_ = self.classifier.fit(X, y)
		#classifier = LogisticRegression(penalty=self.penalty, C = self.C)
		#classifier.fit(X, y)

		y_preds = self.classifier.predict(new_embeddings)

		for i_tile in range(nb_new):

			y_pred = y_preds[i_tile]

			guesses[i_tile].append(y_pred)



		avg_guesses = np.median(guesses, axis=-1)

		final_guesses = np.round(avg_guesses)

		t2 = time.time()

		#print("\tSOLVE 1:", t2 - t1)


		if return_int:
		
			return final_guesses

		else:

			sides = []

			for guess in final_guesses:

				if guess == 0:
					side = "left"
				elif guess == 1:
					side = "right"

				sides.append(side)
			
			
			return sides