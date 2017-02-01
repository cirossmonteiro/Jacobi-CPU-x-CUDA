import os, time, matplotlib.pyplot as plt, cPickle as pk

fn = "cpugpu"
fh = open(fn,'wb')
fh.close()

def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""

maxiter = 100
maxerror = 0.01
all_seq, all_par = {}, {}
dim = 2

try:
    
    while True:

        mode = "seq"
        cmd = "./main %s %d %d %lf" %(mode, dim, maxiter, maxerror)
        start = time.time()
        os.system(cmd)
        end = time.time()
        all_seq[dim] = end - start

        mode = "par"
        cmd = "./main %s %d %d %lf" %(mode, dim, maxiter, maxerror)
        start = time.time()
        os.system(cmd)
        end = time.time()
        all_par[dim] = end - start

        print dim, ", ",
        
        if not dim%100:
            os.remove(fn)
            fh = open(fn,'wb')
            pk.dump([all_seq, all_par], fh)
            fh.close()
            print "saved\n"

        dim += 1

except:
        
    plt.plot(all_seq.keys(), all_seq.values(), 'b-')
    plt.plot(all_par.keys(), all_par.values(), 'r-')
    plt.savefig(fn+'.png')
    plt.show()
    
