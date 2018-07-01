import argparse
import keras
import numpy as np
import pandas
import pickle
from keras import backend as K

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('loadmodel', type=str)
    #parser.add_argument('datafile', type=str)
    parser.add_argument('--nfile', type=str,default='')
    parser.add_argument('--indexfile', type=str, required=True)
    parser.add_argument('--testfile', type=str, default='book_ratings_test.csv')
    parser.add_argument('--outfile', type=str, default='out.csv')

    args = parser.parse_args()
    print(args)

    with open(args.indexfile, 'rb') as f:
        index_dict = pickle.load(f)
    user2index = index_dict['user2index']
    book2index = index_dict['book2index']

    rate_table = pandas.read_csv(args.testfile)

    user_xs = []
    users = {}
    unknown_users = {}
    #known_list = [ True for i in rate_table['User-ID'].values ]
    for idx, i in enumerate(rate_table['User-ID'].values):
        users[i] = None
        if i in user2index:
            user_xs.append(user2index[i])
        else:
            #print("user not found:", i)
            unknown_users[i] = None
            user_xs.append(len(user2index))
            #known_list[idx] = False
    user_xs = np.array(user_xs, dtype=np.int).reshape((-1, 1))
    book_xs = []
    books = {}
    unknown_books = {}
    for idx, i in enumerate(rate_table['ISBN'].values):
        books[i] = None
        if i in book2index:
            book_xs.append(book2index[i])
        else:
            #print("book not found:", i)
            unknown_books[i] = None
            book_xs.append(len(book2index))
            #known_list[idx] = False
    book_xs = np.array(book_xs, dtype=np.int).reshape((-1, 1))

    print("")
    print("num of total user:\t", len(users))
    print("num of no ranking user:\t", len(unknown_users))
    print("")
    print("num of total book:\t", len(books))
    print("num of no ranking book:\t", len(unknown_books))
    print("")

    count_table = np.zeros((2,2), dtype=int) # user X book
    for i in range(len(rate_table['ISBN'].values)):
        count_table[int(rate_table['User-ID'].values[i] not in unknown_users)][int(rate_table['ISBN'].values[i] not in unknown_books)] += 1
    print(count_table)
    print("")
    if args.nfile == '':
        rating_mean = 0
        rating_std = 1
    else:
        with open(args.nfile, 'rb') as f:
            rating_mean, rating_std = pickle.load(f)

    model = keras.models.load_model(args.loadmodel)
    layer_dict = { layer.get_config()['name']:layer for layer in model.layers }
    book_emb = layer_dict['book_emb'].get_weights()[0]
    user_emb = layer_dict['user_emb'].get_weights()[0]
    book_bias = layer_dict['book_bias'].get_weights()[0]
    user_bias = layer_dict['user_bias'].get_weights()[0]
    assert(book_emb.shape[0] == len(book2index)+1)
    assert(user_emb.shape[0] == len(user2index)+1)
    book_emb[-1] = np.mean(book_emb[:-1], axis=0)
    user_emb[-1] = np.mean(user_emb[:-1], axis=0)
    book_bias[-1] = np.mean(book_bias[:-1], axis=0)
    user_bias[-1] = np.mean(user_bias[:-1], axis=0)
    model.get_layer('book_emb').set_weights([book_emb])
    model.get_layer('user_emb').set_weights([user_emb])
    model.get_layer('book_bias').set_weights([book_bias])
    model.get_layer('user_bias').set_weights([user_bias])

    ys = model.predict([user_xs, book_xs], verbose=1, batch_size=256).reshape(-1)
    ys = ys*rating_std + rating_mean
    ys = np.rint(ys).astype(int)

    f = open(args.outfile, "w")
    #f.write("TestDataID,Rating\n")
    for i in range(len(ys)):
        #f.write(str(ans_id[i])+","+str(ys[i])+"\n")
        f.write(str(ys[i])+"\n")
    f.close()

