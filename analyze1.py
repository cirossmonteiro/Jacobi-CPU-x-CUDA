import cPickle as pk
import matplotlib.pyplot as plt

fn = "cpugpu"
fh = open(fn, 'rb')
data = pk.load(fh)
fh.close()

all_seq,all_par = data

plt.plot(all_seq.keys(), all_seq.values(), 'b-')
plt.plot(all_par.keys(), all_par.values(), 'r-')
plt.savefig(fn+'.png')
plt.show()
