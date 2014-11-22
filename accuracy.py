import time
import math
import sys

usage = "filename labels_1 labels_2 feature_index"

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def accuracy(s1,s2,threshold,feature_index):
    line_count =  file_len(s1)
    f1 = open(s1,'r')
    f2 = open(s2,'r')
    count = 0
    try:
        for i in xrange(line_count):
            a = float(f1.readline())
            b = float(f2.readline())
            if((a<=threshold and b<=threshold)or(a>threshold and b>threshold)):
                f3.write("1")
                count = count + 1
            else:
                f3.write("0")
        if(float(feature_index)%4!=0):
            print("%0.2f\t"%((count*1.0/line_count)*100.0))
        else:
            print("%0.2f\t\n"%((count*1.0/line_count)*100.0))
    except IOError:        
            print("Error: IOError at count %d "%count)
    f1.close()
    f2.close()

def main():
    if (len(sys.argv)!=4):
        print usage
    else:
        accuracy(sys.argv[1],sys.argv[2],4.5,sys.argv[3])

if __name__ == "__main__":main()