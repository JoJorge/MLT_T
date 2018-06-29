import numpy as np
import pickle
import pandas
import pdb
from sklearn import linear_model

emb_path = "../data/emb.pickle"
normalize_path = "../data/nor.pickle"
user_path = "../data/users_pre.csv"
book_path = '../data/books.csv'
split_rating_path = '../data/train_split.csv'
imp_rating_path = '../data/implicit_ratings.csv'
users_train_rating_path = '../data/user_valid.csv'

class MF_model:
    def __init__(self, user2emb, book2emb, user2bias, book2bias, rating_mean, rating_std):
        self.user2emb = user2emb
        self.book2emb = book2emb
        self.user2bias = user2bias
        self.book2bias = book2bias
        self.rating_mean = rating_mean
        self.rating_std = rating_std
        self.has_model2 = False
        self.user_avg_emb = (np.average(np.array(list(self.user2emb.values())), axis = 0), np.average(list(self.user2bias.values())))
        self.book_avg_emb = (np.average(np.array(list(self.book2emb.values())), axis = 0), np.average(list(self.book2bias.values())))
    def set_info(self, users_info, books_info, ratings, imp_ratings):
        self.users_info = users_info
        self.books_info = books_info
        self.ratings = ratings
        self.imp_ratings = imp_ratings
    def predict(self, user_id, book_id):
        user_emb = get_user_emb(user_id)
        book_emb = get_book_emb(book_id)
        normalized_rating = np.dot(self.user2emb[user_id], self.book2emb[book_id]) + self.user2bias[user_id] + self.book2bias[book_id]
        rating = normalized_rating*self.rating_std + self.rating_mean
        return rating
    def get_avg_emb(self):
        self.country_emb, self.age_emb, self.reading_users_emb, self.imp_reading_users_emb = {}, {}, {}, {}
        self.author_emb, self.publisher_emb, self.pub_year_emb, self.class_emb, self.read_books_emb, self.imp_read_books_emb = {}, {}, {}, {}, {}, {}
        # countries
        countries = list(set(self.users_info.Location.tolist()))
        for con in countries:
            users = [x for x in self.users_info['User-ID'][self.users_info.Location == con] if x in self.user2emb]
            self.country_emb.update({con: self.get_avg_from_user_IDs(users)})
        # range of age
        ages = list(set(self.users_info.Age.tolist()))
        for age in ages:
            if age == 'unknown':
                self.age_emb.update({age: self.user_avg_emb})
                continue
            users = [x for x in self.users_info['User-ID'][self.users_info.Age == age] if x in self.user2emb]
            user_embs = np.array([self.user2emb[x] for x in users])
            user_biases = np.array([self.user2bias[x] for x in users])
            self.age_emb.update({age: self.get_avg_from_user_IDs(users)})
        # users reading given book
        # too large, so only init unknown
        self.reading_users_emb.update({'unknown': self.user_avg_emb})
        # implicit users reading given book
        self.imp_reading_users_emb.update({'unknown': self.user_avg_emb})
        
        # --TODO--
        # author
        # publisher
        # range of publish year
        # classification
        # --------
        # books read by given user
        self.read_books_emb.update({'unknown': self.book_avg_emb})
        # implicit books read by given user
        self.imp_read_books_emb.update({'unknown': self.book_avg_emb})
    def train_users_weights(self, train_ratings):
        x, y = [], []
        for i, row in train_ratings.iterrows():
            user_id, book_id = row[0], row[1]
            user_info = self.users_info[self.users_info['User-ID'] == user_id].iloc[0,:]
            user_embs = [self.country_emb[user_info.Location], self.age_emb[user_info.Age], self.get_reading_user_emb(book_id), self.get_imp_reading_user_emb(book_id)]
            book_emb = (self.book2emb[book_id], self.book2bias[book_id])
            x.append([np.dot(v[0], book_emb[0]) + v[1] for v in user_embs])
            y.append(row[2] - book_emb[1])
        # TODO: TRAIN
        reg = linear_model.LinearRegression(fit_intercept = False)
        reg.fit(x, y)
        self.user_weights = reg.coef_
        
    def get_user_emb(self, user_id):
        if user_id in self.user2emb:
            return (self.user2emb[user_id], self.user2bias[user_id])
        if self.has_model2 == False:
            return self.user_avg_emb
        else:
            # --TODO--
            # use the weight and other info to get embedding
            # --------
            return None
    def get_book_emb(self, book_id):
        if book_id in self.book2emb:
            return (self.book2emb[book_id], self.book2bias[book_id])
        if self.has_model2 == False:
            return self.book_avg_emb
        else:
            # --TODO--
            # use the weight and other info to get embedding
            # --------
            return None
    def get_avg_from_user_IDs(self, users):
        user_embs = np.array([self.user2emb[x] for x in users])
        user_biases = np.array([self.user2bias[x] for x in users])
        return (np.average(user_embs, axis=0), np.average(user_biases))
    def get_reading_user_emb(self, book_id):
        if book_id not in self.reading_users_emb:
            users = self.ratings['User-ID'][self.ratings.ISBN == book_id]
            users = [x for x in users if x in self.user2emb]
            if len(users) == 0:
                self.reading_users_emb.update({book_id: self.user_avg_emb})
            else:
                self.reading_users_emb.update({book_id: self.get_avg_from_user_IDs(users)})
        return self.reading_users_emb[book_id]
    def get_imp_reading_user_emb(self, book_id):
        if book_id not in self.imp_reading_users_emb:
            users = self.imp_ratings['User-ID'][(self.imp_ratings.ISBN == book_id)].tolist()
            users = [x for x in users if x in self.user2emb]
            if len(users) == 0:
                self.imp_reading_users_emb.update({book_id: self.user_avg_emb})
            else:
                self.imp_reading_users_emb.update({book_id: self.get_avg_from_user_IDs(users)})
        return self.imp_reading_users_emb[book_id]
            
def read_info(user_path, book_path):
    users_info = pandas.read_csv(user_path)
    books_info = pandas.read_csv(book_path)
    split_ratings = pandas.read_csv(split_rating_path)
    imp_ratings = pandas.read_csv(imp_rating_path)
    return users_info, books_info, split_ratings, imp_ratings

def read_model(emb_path, normalize_path):
    with open(emb_path, 'rb') as f:
        emb_dict = pickle.load(f)
    user2emb = emb_dict['user2emb']
    book2emb = emb_dict['book2emb']
    user2bias = emb_dict['user2bias']
    book2bias = emb_dict['book2bias']

    with open(normalize_path, 'rb') as f:
        rating_mean, rating_std = pickle.load(f)
    return MF_model(user2emb, book2emb, user2bias, book2bias, rating_mean, rating_std)
    
if __name__ == '__main__':
    mf_model = read_model(emb_path, normalize_path)
    users_info, books_info, split_ratings, imp_ratings = read_info(user_path, book_path)
    mf_model.set_info(users_info, books_info, split_ratings, imp_ratings)
    pdb.set_trace()
    mf_model.get_avg_emb()
    users_train_ratings = pandas.read_csv(users_train_rating_path)
    mf_model.train_users_weights(users_train_ratings)
    user_id = '019be1539c'
    book_id = '1841954241'

    rating = mf_model.predict(user_id, book_id)

    print("predict rating for user '%s' and book '%s' = %f"%(user_id, book_id, rating))
