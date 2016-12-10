import cPickle as pickle
import gzip

class Snapshotter:

    def __init__(self):
        pass

    @classmethod
    def snapshot(self, fname, obj):
        #store the object
        f = gzip.open(fname, 'wb')
        pickle.dump(obj, f)
        f.close()

    @classmethod
    def load(self, fname):
        #restore the object
        f = gzip.open(fname, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj
