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
user2info_path = data_path + 'user2info.pickle'
book_path = data_path + 'books_pre.csv'
book2info_path = data_path + 'book2info.pickle'
split_rating_path = data_path + 'train_split.csv'
imp_rating_path = data_path + 'implicit_ratings.csv'
users_train_rating_path = data_path + 'user_valid.csv'
books_train_rating_path = data_path + 'book_valid.csv'
model_path = data_path + 'mf_model.pickle'
avg_embs_path = data_path + 'avg_embs.pickle'
weights_path = data_path + 'weights.pickle'
test_path = data_path + 'book_ratings_test.csv'
predict_path = data_path + 'predict.csv'

class MF_model:
    def __init__(self, user2emb, book2emb, user2bias, book2bias, rating_mean, rating_std):
        self.user2emb = user2emb
        self.book2emb = book2emb
        self.user2bias = user2bias
        self.book2bias = book2bias
        self.rating_mean = rating_mean
        self.rating_std = rating_std
        self.country_emb, self.age_emb, self.reading_users_emb, self.imp_reading_users_emb = {}, {}, {}, {}
        self.author_emb, self.publisher_emb, self.pub_year_emb, self.class_emb, self.read_books_emb, self.imp_read_books_emb = {}, {}, {}, {}, {}, {}
        self.has_user_weights = False
        self.has_book_weights = False
        self.user_weights = np.zeros(4)
        self.book_weights = np.zeros(6)
        self.user_avg_emb = (np.average(np.array(list(self.user2emb.values())), axis = 0), np.average(list(self.user2bias.values())))
        self.book_avg_emb = (np.average(np.array(list(self.book2emb.values())), axis = 0), np.average(list(self.book2bias.values())))
    def set_info(self, users_info, books_info, user2info, book2info):
        self.users_info = users_info
        self.user2info = user2info
        self.books_info = books_info
        self.book2info = book2info
    def reset_info_dict(self):
        self.user2info = {row['User-ID']: row for i, row in self.users_info.iterrows()}
        self.book2info = {row.ISBN: row for i, row in self.books_info.iterrows()}
        with open(user2info_path, 'wb') as f:
            pickle.dump(self.user2info, f, protocol = pickle.HIGHEST_PROTOCOL)
        with open(book2info_path, 'wb') as f:
            pickle.dump(self.book2info, f, protocol = pickle.HIGHEST_PROTOCOL)
    def set_ratings(self, ratings, imp_ratings):
        self.ratings = ratings
        self.imp_ratings = imp_ratings
    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    def load_model(model_path):
        mf_model = None
        with open(model_path, 'rb') as f:
            mf_model = pickle.load(f)
        return mf_model
    def predict(self, user_id, book_id):
        user_emb = self.get_user_emb(user_id, book_id)
        book_emb = self.get_book_emb(user_id, book_id)
        normalized_rating = np.dot(user_emb[0], book_emb[0]) + user_emb[1] + book_emb[1]
        rating = normalized_rating*self.rating_std + self.rating_mean
        if type(rating) == np.ndarray:
            rating = rating[0]
        rating = min(max(rating, 0), 10)
        return int(round(rating))
    def save_large_avg_emb(self):
        avg_embs = {'reading_users': self.reading_users_emb, 'imp_reading_users': self.imp_reading_users_emb}
        avg_embs.update({'read_books': self.read_books_emb, 'imp_read_books': self.imp_read_books_emb})
        with open(avg_embs_path, 'wb') as f:
            pickle.dump(avg_embs, f, protocol=pickle.HIGHEST_PROTOCOL)
    def load_large_avg_emb(self, avg_embs_path):
        avg_embs = {}
        with open(avg_embs_path, 'rb') as f:
            avg_embs = pickle.load(f)
        self.reading_users_emb = avg_embs['reading_users']
        self.imp_reading_users_emb = avg_embs['imp_reading_users']
        self.read_books_emb = avg_embs['read_books']
        self.imp_read_books_emb = avg_embs['imp_read_books']
    def get_avg_emb(self):
        print('Getting users avg embedding....')
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
            self.age_emb.update({age: self.get_avg_from_user_IDs(users)})
        # users reading given book
        # too large, so only init unknown
        self.reading_users_emb.update({'unknown': self.user_avg_emb})
        # implicit users reading given book
        self.imp_reading_users_emb.update({'unknown': self.user_avg_emb})
        
        print('Getting books avg embedding....')
        # author 
        for book_id in self.ratings.ISBN:
            if book_id not in self.book2info:
                continue
            if self.book2info[book_id]['Book-Author'] in self.author_emb:
                continue
            author = self.book2info[book_id]['Book-Author']
            if author == 'only_1':
                self.author_emb.update({author: self.book_avg_emb})
                continue
            books = [x for x in self.books_info.ISBN[self.books_info['Book-Author'] == author] if x in self.book2emb]
            self.author_emb.update({author: self.get_avg_from_book_IDs(books)})
        # publisher
        for book_id in self.ratings.ISBN:
            if book_id not in self.book2info:
                continue
            if self.book2info[book_id].Publisher in self.publisher_emb:
                continue
            publisher = self.book2info[book_id].Publisher
            if publisher == 'only_1':
                self.publisher_emb.update({publisher: self.book_avg_emb})
                continue
            books = [x for x in self.books_info.ISBN[self.books_info.Publisher == publisher] if x in self.book2emb]
            self.publisher_emb.update({publisher: self.get_avg_from_book_IDs(books)})
        # range of publish year
        pub_years = list(set(self.books_info['Year-Of-Publication'].tolist()))
        for pub_year in pub_years:
            books = [x for x in self.books_info.ISBN[self.books_info['Year-Of-Publication'] == pub_year] if x in self.book2emb]
            self.pub_year_emb.update({pub_year: self.get_avg_from_book_IDs(books)})
        # --TODO--
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
            if (i+1) % 50 == 0:
                print('.', end='', flush = True)
            if (i+1) % 1000 == 0:
                print('', flush = True)
                
        # TODO: TRAIN with MAE
        reg = linear_model.LinearRegression(fit_intercept = False)
        reg.fit(x, y)
        self.user_weights = reg.coef_.flatten()
        print('Done')
        y_g = reg.predict(x)
        error = mean_absolute_error(y_g, y) * self.rating_std
        print('MAE of user_valid:', error, '\n')
    def train_books_weights(self, train_ratings):
        print('Training book weights....')
        self.has_book_weights = True
        x, y = [], []
        for i, row in train_ratings.iterrows():
            user_id, book_id = row[0], row[1]
            user_emb = (self.user2emb[user_id], self.user2bias[user_id])
            book_embs = self.get_book_embs(user_id, book_id)
            x.append([np.dot(user_emb[0], v[0]) + v[1] for v in book_embs])
            y.append((row[2] - self.rating_mean) / self.rating_std - user_emb[1])
            if (i+1) % 50 == 0:
                print('.', end='', flush = True)
            if (i+1) % 1000 == 0:
                print('', flush = True)
                
        # TODO: TRAIN with MAE
        reg = linear_model.LinearRegression(fit_intercept = False)
        reg.fit(x, y)
        self.book_weights = reg.coef_.flatten()
        print('Done')
        y_g = reg.predict(x)
        error = mean_absolute_error(y_g, y) * self.rating_std
        print('MAE of book_valid:', error, '\n')
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
            book_embs = self.get_book_embs(user_id, book_id)
            book_emb = np.average([x[0] for x in book_embs], axis = 0, weights = self.book_weights)
            book_bias = np.average([x[1] for x in book_embs], weights = self.book_weights)
            return (book_emb, book_bias)
    def get_user_embs(self, user_id, book_id):
        user_embs = []
        if user_id not in self.user2info:
            user_embs = [self.user_avg_emb] * 2
        else:
            user_info = self.user2info[user_id]
            user_embs = [self.country_emb[user_info.Location], self.age_emb[user_info.Age]]
        user_embs.append(self.get_reading_users_emb(book_id))
        user_embs.append(self.get_imp_reading_users_emb(book_id))
        return user_embs
    def get_book_embs(self, user_id, book_id):
        book_embs = []
        if book_id not in self.book2info:
            book_embs = [self.book_avg_emb] * 3
        else:
            book_info = self.book2info[book_id]
            if book_info['Book-Author'] in self.author_emb:
                book_embs.append(self.author_emb[book_info['Book-Author']])
            else:
                book_embs.append(self.book_avg_emb)
            if book_info.Publisher in self.publisher_emb:
                book_embs.append(self.publisher_emb[book_info.Publisher])
            else:
                book_embs.append(self.book_avg_emb)
            # book_embs = [self.author_emb[book_info['Book-Author']], self.publisher_emb[book_info.Publisher]]
            #book_embs.extend([self.pub_year_emb[book_info['Year-Of-Publication']], self.class_emb[book_info.Classification]])
            book_embs.append(self.pub_year_emb[book_info['Year-Of-Publication']])
        book_embs.append(self.get_read_books_emb(user_id))
        book_embs.append(self.get_imp_read_books_emb(user_id))
        return book_embs
    def get_avg_from_user_IDs(self, users):
        if len(users) == 0:
            return self.user_avg_emb
        user_embs = np.array([self.user2emb[x] for x in users])
        user_biases = np.array([self.user2bias[x] for x in users])
        return (np.average(user_embs, axis=0), np.average(user_biases))
    def get_avg_from_book_IDs(self, books):
        if len(books) == 0:
            return self.book_avg_emb
        book_embs = np.array([self.book2emb[x] for x in books])
        book_biases = np.array([self.book2bias[x] for x in books])
        return (np.average(book_embs, axis=0), np.average(book_biases))
    def get_reading_users_emb(self, book_id):
        if book_id not in self.reading_users_emb:
            users = self.ratings['User-ID'][self.ratings.ISBN == book_id]
            users = [x for x in users if x in self.user2emb]
            self.reading_users_emb.update({book_id: self.get_avg_from_user_IDs(users)})
        return self.reading_users_emb[book_id]
    def get_imp_reading_users_emb(self, book_id):
        if book_id not in self.imp_reading_users_emb:
            users = self.imp_ratings['User-ID'][(self.imp_ratings.ISBN == book_id)].tolist()
            users = [x for x in users if x in self.user2emb]
            self.imp_reading_users_emb.update({book_id: self.get_avg_from_user_IDs(users)})
        return self.imp_reading_users_emb[book_id]
    def get_read_books_emb(self, user_id):
        if user_id not in self.read_books_emb:
            books = self.ratings.ISBN[self.ratings['User-ID'] == user_id]
            books = [x for x in books if x in self.book2emb]
            self.read_books_emb.update({user_id: self.get_avg_from_book_IDs(books)})
        return self.read_books_emb[user_id]
    def get_imp_read_books_emb(self, user_id):
        if user_id not in self.imp_read_books_emb:
            books = self.imp_ratings.ISBN[(self.imp_ratings['User-ID'] == user_id)].tolist()
            books = [x for x in books if x in self.book2emb]
            self.imp_read_books_emb.update({user_id: self.get_avg_from_book_IDs(books)})
        return self.imp_read_books_emb[user_id]
            
def read_info(user_path, book_path, user2info_path, book2info_path):
    users_info = pandas.read_csv(user_path)
    books_info = pandas.read_csv(book_path)
    user2info = {}
    with open(user2info_path, 'rb') as f:
        user2info = pickle.load(f)
    book2info = {}
    with open(book2info_path, 'rb') as f:
        book2info = pickle.load(f)
    split_ratings = pandas.read_csv(split_rating_path)
    imp_ratings = pandas.read_csv(imp_rating_path)
    return users_info, books_info, user2info, book2info
def read_rating(rating_path, imp_rating_path):
    ratings = pandas.read_csv(rating_path)
    imp_ratings = pandas.read_csv(imp_rating_path)
    return ratings, imp_ratings

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
    '''
    print('Reading model and info....')
    mf_model = read_model(emb_path, normalize_path)
    mf_model.set_info(*read_info(user_path, book_path, user2info_path, book2info_path))
    mf_model.set_ratings(*read_rating(split_rating_path, imp_rating_path))
    pdb.set_trace()
    
    print('Getting average embedding and load preprocess data....')
    mf_model.get_avg_emb()
    mf_model.save_model(model_path)
    pdb.set_trace()
    '''
    print('Reading saved model....')
    mf_model = MF_model.load_model(model_path)
    pdb.set_trace()
    
    # user train
    '''
    users_train_ratings = pandas.read_csv(users_train_rating_path)
    mf_model.train_users_weights(users_train_ratings)
    pdb.set_trace()
    #'''
    
    # book train
    '''
    books_train_ratings = pandas.read_csv(books_train_rating_path)
    mf_model.train_books_weights(books_train_ratings)
    pdb.set_trace()
    #'''
    
    '''
    print('Saving....')
    mf_model.save_large_avg_emb()
    mf_model.save_model(model_path)
    pdb.set_trace()
    '''
    
    # predict test data
    print('Predicting....')
    test_rating = pandas.read_csv(test_path)
    y_g = []
    for i, row in test_rating.iterrows():
        y_g.append(mf_model.predict(row[0], row[1]))
        if (i+1) % 200 == 0:
            print('.', end='', flush = True)
        if (i+1) % 10000 == 0:
            print('', flush = True)
    pdb.set_trace()
    with open(predict_path, 'w') as f:
        for y in y_g:
            f.write('%d\n'%(y))
    
    print('Saving again....')
    mf_model.save_large_avg_emb()
    mf_model.save_model(model_path)
    pdb.set_trace()
    