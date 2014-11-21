import time
import sys
import numpy
import vector
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.cross_validation import train_test_split

usage = "filename features_file labels_file output_file"

if __name__ == "__main__":

	if (len(sys.argv)!=5):
		print usage
	else:
		file_x = sys.argv[1]
		file_y = sys.argv[2]
		file_out = sys.argv[3]
		split_seed = sys.argv[4]

		X = numpy.genfromtxt(file_x, delimiter=' ')
		y = numpy.genfromtxt(file_y, delimiter=' ')

		# Split the data into training/testing sets
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=split_seed)
		
		# Bayesian Ridge Regression
		clf = BayesianRidge(compute_score=True)
		clf.fit(X, y)
		y_predict=clf.predict(X_test)
		numpy.savetxt(file_out, y_predict)
