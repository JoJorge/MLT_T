import argparse
import keras
import numpy as np
import pandas
import pickle
from keras import backend as K

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('loadmodel', type=str)
    parser.add_argument('--nfile', type=str, default='')
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
    unknow_users = {}
    for i in rate_table['User-ID'].values:
        users[i] = None
        if i in user2index:
            user_xs.append(user2index[i])
        else:
            #print("user not found:", i)
            unknow_users[i] = None
            user_xs.append(0)
    user_xs = np.array(user_xs, dtype=np.int).reshape((-1, 1))
    book_xs = []
    books = {}
    unknow_books = {}
    for i in rate_table['ISBN'].values:
        books[i] = None
        if i in book2index:
            book_xs.append(book2index[i])
        else:
            #print("book not found:", i)
            unknow_books[i] = None
            book_xs.append(0)
    book_xs = np.array(book_xs, dtype=np.int).reshape((-1, 1))

    print("")
    print("num of total user:\t", len(users))
    print("num of no ranking user:\t", len(unknow_users))
    print("")
    print("num of total book:\t", len(books))
    print("num of no ranking book:\t", len(unknow_books))
    print("")

    count_table = np.zeros((2,2), dtype=int) # user X book
    for i in range(len(rate_table['ISBN'].values)):
        count_table[int(rate_table['User-ID'].values[i] not in unknow_users)][int(rate_table['ISBN'].values[i] not in unknow_books)] += 1
    print(count_table)
    print("")
    if args.nfile == '':
        rating_mean = 0
        rating_std = 1
    else:
        with open(args.nfile, 'rb') as f:
            rating_mean, rating_std = pickle.load(f)

    model = keras.models.load_model(args.loadmodel)
    ys = model.predict([user_xs, book_xs], verbose=1, batch_size=256).reshape(-1)
    ys = ys*rating_std + rating_mean
    ys = np.rint(ys).astype(int)

    f = open(args.outfile, "w")
    #f.write("TestDataID,Rating\n")
    for i in range(len(ys)):
        #f.write(str(ans_id[i])+","+str(ys[i])+"\n")
        f.write(str(ys[i])+"\n")
    f.close()

