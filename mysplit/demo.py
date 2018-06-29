import numpy as np
import pickle

if __name__ == '__main__':
    with open("emb.pickle", 'rb') as f:
        emb_dict = pickle.load(f)
    user2emb = emb_dict['user2emb']
    book2emb = emb_dict['book2emb']
    user2bias = emb_dict['user2bias']
    book2bias = emb_dict['book2bias']

    with open("nor.pickle", 'rb') as f:
        rating_mean, rating_std = pickle.load(f)

    user_id = '019be1539c'
    book_id = '1841954241'

    normalized_rating = np.dot(user2emb[user_id], book2emb[book_id]) + user2bias[user_id] + book2bias[book_id]
    rating = normalized_rating*rating_std + rating_mean

    print("predict rating for user '%s' and book '%s' = %f"%(user_id, book_id, rating))
