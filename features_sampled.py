import cPickle
import os.path
from multiprocessing import Pool
import sys

usage = "filename out_file"

def generate_features(fout_file):
	nLabel, nTrial, nUser, nChannel, nTime  = 4, 40, 32, 40, 8064
	#new_array = [[[None] *w for i in range(h)] for j in range(l)]
	print "Program started"+"\n"
	fout_data = open(fout_file,'w')
	for i in range(nUser):#4, 40, 32, 40, 8064
		if(i%8 == 0):
			if i < 10:
				name = '%0*d' % (2,i+1)
			else:
				name = i+1
			fname = "data/raw/s"+str(name)+".dat"
			x = cPickle.load(open(fname, 'rb'))
			print fname
			for tr in range(nTrial):
				if(tr%1 == 0):
					for dat in range(nTime):
						if(dat%64 == 0):
							for ch in range(nChannel):
								#fout_data.write(str(ch+1) + " ");
								fout_data.write(str(x['data'][tr][ch][dat]) + " ");
					fout_data.write("\n");
	fout_data.close()

if __name__ == "__main__":
	if (len(sys.argv)!=2):
		print usage
	else:
		generate_features(sys.argv[1])