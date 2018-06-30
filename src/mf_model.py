import numpy as np
import pickle
import pandas
import pdb
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

data_path = '../data/'
emb_path = data_path + 'emb.pickle'
normalize_path = data_path + 'nor.pickle'
user_path = data_path + 'users_pre.csv'
book_path = data_path + 'books.csv'
split_rating_path = data_path + 'train_split.csv'
imp_rating_path = data_path + 'implicit_ratings.csv'
users_train_rating_path = data_path + 'user_valid.csv'
reading_users_emb_path = data_path + 'reading_users_emb.pickle'
imp_reading_users_emb_path = data_path + 'imp_reading_users_emb.pickle'
weights_path = data_path + 'weights.pickle'

class MF_model:
    def __init__(self, user2emb, book2emb, user2bias, book2bias, rating_mean, rating_std):
        self.user2emb = user2emb
        self.book2emb = book2emb
        self.user2bias = user2bias
        self.book2bias = book2bias
        self.rating_mean = rating_mean
        self.rating_std = rating_std
        self.has_user_weights = False
        self.has_book_weights = False
        self.user_weights = np.zeros(4)
        self.book_weights = np.zeros(6)
        self.user_avg_emb = (np.average(np.array(list(self.user2emb.values())), axis = 0), np.average(list(self.user2bias.values())))
        self.book_avg_emb = (np.average(np.array(list(self.book2emb.values())), axis = 0), np.average(list(self.book2bias.values())))
    def set_info(self, users_info, books_info, ratings, imp_ratings):
        self.users_info = users_info
        self.books_info = books_info
        self.ratings = ratings
        self.imp_ratings = imp_ratings
    def predict(self, user_id, book_id):
        user_emb = self.get_user_emb(user_id, book_id)
        book_emb = self.get_book_emb(user_id, book_id)
        normalized_rating = np.dot(user_emb[0], book_emb[0]) + user_emb[1] + book_emb[1]
        rating = normalized_rating*self.rating_std + self.rating_mean
        return rating
    def save_large_avg_emb(self):
        with open(reading_users_emb_path, 'wb') as f:
            pickle.dump(self.reading_users_emb, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(imp_reading_users_emb_path, 'wb') as f:
            pickle.dump(self.imp_reading_users_emb, f, protocol=pickle.HIGHEST_PROTOCOL)
    def load_large_avg_emb(self):
        with open(reading_users_emb_path, 'rb') as f:
            self.reading_users_emb = pickle.load(f)
        with open(imp_reading_users_emb_path, 'rb') as f:
            self.imp_reading_users_emb = pickle.load(f)
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
        print('Training user weights....')
        self.has_user_weights = True
        x, y = [], []
        for i, row in train_ratings.iterrows():
            user_id, book_id = row[0], row[1]
            user_embs = self.get_user_embs(user_id, book_id)
            book_emb = (self.book2emb[book_id], self.book2bias[book_id])
            x.append([np.dot(v[0], book_emb[0]) + v[1] for v in user_embs])
            y.append((row[2] - self.rating_mean) / self.rating_std - book_emb[1])
            if (i+1) % 20 == 0:
                print('.', end='', flush = True)
            if (i+1) % 500 == 0:
                print('', flush = True)
                
        # TODO: TRAIN with MAE
        reg = linear_model.LinearRegression(fit_intercept = False)
        reg.fit(x, y)
        self.user_weights = reg.coef_.flatten()
        print('Done')
        y_g = reg.predict(x)
        error = mean_absolute_error(y_g, y) * self.rating_std
        print('MAE of user_valid:', error, '\n')
    def save_weights(self):
        with open(weights_path, 'wb') as f:
            pickle.dump((self.user_weights, self.book_weights), f, pickle.HIGHEST_PROTOCOL)
    def load_weights(self):
        with open(weights_path, 'rb') as f:
            self.user_weights, self.book_weights = pickle.load(f)
        self.has_user_weights = (np.sum(np.absolute(self.user_weights)) != 0)
        self.has_book_weights = (np.sum(np.absolute(self.book_weights)) != 0)
    def get_user_emb(self, user_id, book_id):
        if user_id in self.user2emb:
            return (self.user2emb[user_id], self.user2bias[user_id])
        if self.has_user_weights == False:
            return self.user_avg_emb
        else:
            user_embs = self.get_user_embs(user_id, book_id)
            user_emb = np.average([x[0] for x in user_embs], axis = 0, weights = self.user_weights)
            user_bias = np.average([x[1] for x in user_embs], weights = self.user_weights)
            return (user_emb, user_bias)
    def get_book_emb(self, user_id, book_id):
        if book_id in self.book2emb:
            return (self.book2emb[book_id], self.book2bias[book_id])
        if self.has_book_weights == False:
            return self.book_avg_emb
        else:
            # --TODO--
            # use the weight and other info to get embedding
            # --------
            return None
    def get_user_embs(self, user_id, book_id):
        user_info = self.users_info[self.users_info['User-ID'] == user_id].iloc[0,:]
        user_embs = [self.country_emb[user_info.Location], self.age_emb[user_info.Age]]
        user_embs.append(self.get_reading_users_emb(book_id))
        user_embs.append(self.get_imp_reading_users_emb(book_id))
        return user_embs
    def get_avg_from_user_IDs(self, users):
        user_embs = np.array([self.user2emb[x] for x in users])
        user_biases = np.array([self.user2bias[x] for x in users])
        return (np.average(user_embs, axis=0), np.average(user_biases))
    def get_reading_users_emb(self, book_id):
        if book_id not in self.reading_users_emb:
            users = self.ratings['User-ID'][self.ratings.ISBN == book_id]
            users = [x for x in users if x in self.user2emb]
            if len(users) == 0:
                self.reading_users_emb.update({book_id: self.user_avg_emb})
            else:
                self.reading_users_emb.update({book_id: self.get_avg_from_user_IDs(users)})
        return self.reading_users_emb[book_id]
    def get_imp_reading_users_emb(self, book_id):
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
    
    mf_model.get_avg_emb()
    mf_model.load_large_avg_emb()
    mf_model.load_weights()
    pdb.set_trace()
    
    # user train
    '''
    users_train_ratings = pandas.read_csv(users_train_rating_path)
    mf_model.train_users_weights(users_train_ratings)
    pdb.set_trace()
    #'''
    
    # book train
    # TODO
    
    mf_model.save_weights()
