import pickle

for i in range(2000, 2500):
    filename = './batch_1/' + str(i) + '.pkl'
    with open(filename, 'rb') as f:
        x = pickle.load(f)
        print (i, '..', x[1])