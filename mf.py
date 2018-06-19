import argparse
import os
import numpy as np
import pickle
import keras
from keras.layers import Dense, Dropout, Flatten, Activation, Embedding, Input, Dot, Add
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import regularizers
import pandas
#import tensorflow as tf
#tf.Session(config=tf.ConfigProto(log_device_placement=True))
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


#embedding_size = 80

def get_model(user_num, book_num, embedding_size, no_bias):
    user_in = Input(shape=[1])
    book_in = Input(shape=[1])

    user_emb = Flatten()(Embedding(user_num, embedding_size, input_length=1)(user_in))
    book_emb = Flatten()(Embedding(book_num, embedding_size, input_length=1)(book_in))
    #user_emb = Flatten()(Embedding(user_num, embedding_size, input_length=1, embeddings_regularizer=regularizers.l2(0.0001))(user_in))
    #book_emb = Flatten()(Embedding(book_num, embedding_size, input_length=1, embeddings_regularizer=regularizers.l2(0.0001))(book_in))
    d = Dot(1)([user_emb, book_emb])

    if no_bias:
        model = keras.models.Model([user_in, book_in], d)
    else:
        user_bias = Flatten()(Embedding(user_num, 1, input_length=1)(user_in))
        book_bias = Flatten()(Embedding(book_num, 1, input_length=1)(book_in))
        bias = Add()([user_bias, book_bias])

        out = Add()([d, bias])
        model = keras.models.Model([user_in, book_in], out)
    #model.summary()
    return model

def get_dict(ids):
    id2index = {}
    num = 0
    for s in ids:
        if s not in id2index:
            id2index[s] = num
            num += 1
    return id2index, num
def read_data(filename):
    rate_table = pandas.read_csv(filename)

    user_id = rate_table['User-ID'].values
    user2index, user_num = get_dict(user_id)
    user_xs = np.array([ user2index[i] for i in user_id ], dtype=np.int)

    book_id = rate_table['ISBN'].values
    book2index, book_num = get_dict(book_id)
    book_xs = np.array([ book2index[i] for i in book_id ], dtype=np.int)

    rating = np.array(rate_table['Book-Rating'].values, dtype=np.float32)
    return user2index, book2index, user_xs, book_xs, rating

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default="book_ratings_train.csv")
    #parser.add_argument('--book-file', type=str, default="books.csv")
    #parser.add_argument('--user-file', type=str, default="users.csv")
    parser.add_argument('--num-epoch', type=int, default=10)
    parser.add_argument('--embedding-size', type=int, default=50)
    parser.add_argument('--outdir', type=str, default='modeldir')
    #parser.add_argument('--valid-save', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--train-rate', type=float, default=0.99)
    parser.add_argument('--normalize', action="store_true")
    parser.add_argument('--no-bias', action="store_true")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # f = open(args.book_file, errors='replace')
    # f.readline() #first line
    # book_num = 1 + max([int(line.split("::",maxsplit=1)[0]) for line in f])
    # f.close()

    # f = open(args.user_file, errors='replace')
    # f.readline()
    # user_num = 1 + max([int(line.split("::",maxsplit=1)[0]) for line in f])
    # f.close()

    user2index, book2index, user_xs, book_xs, ys = read_data(args.train_file)
    user_num = len(user2index)
    book_num = len(book2index)
    with open(args.outdir + "/index.pickle", 'wb') as f:
        pickle.dump({"user2index":user2index, "book2index":book2index}, f)

    shuffle_index = np.random.permutation(len(ys))
    user_xs = user_xs[shuffle_index].reshape((-1,1))
    book_xs = book_xs[shuffle_index].reshape((-1,1))
    ys = ys[shuffle_index]#.reshape((-1,1))

    train_num = int(len(ys) * args.train_rate + 0.5)

    user_v = user_xs[train_num:]
    book_v = book_xs[train_num:]
    vys = ys[train_num:]

    user_xs = user_xs[:train_num]
    book_xs = book_xs[:train_num]
    ys = ys[:train_num]

    if args.normalize:
        rating_mean = np.mean(ys)
        rating_std = np.std(ys)
        ys = (ys - rating_mean) / rating_std
        vys = (vys - rating_mean) / rating_std
        prefix = "normlz-"
    else:
        rating_mean = 0
        rating_std = 1
        prefix = ""

    print(rating_mean, rating_std)
    with open(args.outdir + "/nor.pickle", 'wb') as f:
        pickle.dump((rating_mean, rating_std), f)

    model = get_model(user_num, book_num, args.embedding_size, args.no_bias)
    #model.compile(loss='mean_squared_error',
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizers.Adam(),
                  metrics='accuracy'.split(','))

    checkpoint = ModelCheckpoint(
        args.outdir + "/" + prefix + "model-{epoch:02d}.hdf5",
        verbose=0, save_best_only=False, period=2)

    h = model.fit([user_xs, book_xs], ys, batch_size=args.batch_size, epochs=args.num_epoch, callbacks=[checkpoint], validation_data=([user_v, book_v], vys), verbose=1)
