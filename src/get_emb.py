import pdb
import argparse
import keras
import numpy as np
import pandas
import pickle
from keras import backend as K

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('loadmodel', type=str)
    parser.add_argument('--indexfile', type=str, required=True)
    parser.add_argument('--outfile', type=str, default='emb.pickle')
    args = parser.parse_args()

    model = keras.models.load_model(args.loadmodel)

    weight_dict = { layer.get_config()['name']:layer.get_weights() for layer in model.layers }
    book_emb = weight_dict['book_emb'][0]
    user_emb = weight_dict['user_emb'][0]
    book_bias = weight_dict['book_bias'][0]
    user_bias = weight_dict['user_bias'][0]

    with open(args.indexfile, 'rb') as f:
        index_dict = pickle.load(f)
    user2index = index_dict['user2index']
    book2index = index_dict['book2index']

    user2emb = { key:user_emb[index] for key,index in user2index.items() }
    book2emb = { key:book_emb[index] for key,index in book2index.items() }
    user2bias = { key:user_bias[index] for key,index in user2index.items() }
    book2bias = { key:book_bias[index] for key,index in book2index.items() }
    with open(args.outfile, 'wb') as f:
        pickle.dump({"user2emb":user2emb, "book2emb":book2emb, "user2bias":user2bias, "book2bias":book2bias}, f)
    #pdb.set_trace()
