import time
import sys
import numpy
import vector
from bahsic import CBAHSIC

usage = "yolo"

if __name__ == "__main__":

	if (len(sys.argv)<4):
		print usage
	else:
		file_x = sys.argv[1];
		file_y = sys.argv[2];
		file_out = sys.argv[3];
		if (sys.argv==5):
			file_normalized = sys.argv[5]

		X = numpy.genfromtxt(file_x, delimiter=' ')
		y = numpy.genfromtxt(file_y, delimiter=' ')
		
		bahsic = CBAHSIC()
		data_no = 160
		features_tokeep = 5040
		y.shape = (data_no,1)

		# Normalize the labels.
		y = 1.0*y
		tmp_no = numpy.sum(y)
		pno = (data_no + tmp_no) / 2
		nno = (data_no - tmp_no) / 2
		y[y>0] = y[y>0]/pno
		y[y<0] = y[y<0]/nno

		# Normalize the data. 
		m = X.mean(0)
		s = X.std(0)
		X.__isub__(m).__idiv__(s)

		t1 = time.clock()
		tmp = bahsic.BAHSICRaw(X, y, vector.CLinearKernel(), vector.CLinearKernel(), features_tokeep, 0.1)
		t2 = time.clock()
		print "time taken: "+str(t2-t1)
		print '--rank of the features'
		print '--better features towards the end of the list:'
		print tmp

		hsicfeatures= numpy.zeros(shape=(data_no,features_tokeep))
		for i in range(0,data_no):
			for j in range(0,features_tokeep):
				hsicfeatures[i][j] = X[i][tmp[features_tokeep+j]]

		numpy.savetxt(file_out, hsicfeatures)
		if (sys.argv==5):
			numpy.savetxt('original.csv', X)