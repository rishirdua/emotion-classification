import time
import sys
import numpy
import vector
from sklearn.cross_validation import train_test_split

usage = "yolo"

if __name__ == "__main__":

	if (len(sys.argv)!=4):
		print usage
	else:
		file_x = sys.argv[1];
		file_y = sys.argv[2];
		file_y_test = sys.argv[3];

		X = numpy.genfromtxt(file_x, delimiter=' ')
		y = numpy.genfromtxt(file_y, delimiter=' ')
		
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
		numpy.savetxt(file_y_test, y_test)
		numpy.savetxt("data/features_train.dat",X_train)
		numpy.savetxt("data/features_test.dat",X_test)