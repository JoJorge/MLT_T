# python 3.6
# preprocessing user and book data for ML second train

import pandas
import pickle
import pdb

data_path = '../data/'
user_path = data_path + 'users_pre.csv'
book_path = data_path + 'books_pre.csv'
rating_path = data_path + 'book_ratings_train.csv'
imp_rating_path = data_path + 'implicit_ratings.csv'

id2info_path = data_path + 'id2info.pickle'
featre2id_path = data_path + 'f2id.pickle'

users_info = pandas.read_csv(user_path)
books_info = pandas.read_csv(book_path)
ratings = pandas.read_csv(rating_path)
imp_ratings = pandas.read_csv(imp_rating_path)

def get_unique(series):
    return list(set(series.tolist()))

# user/book_id to info
print('Preprocessing id to info....')
user2info = {row['User-ID']: row for i, row in users_info.iterrows()}
book2info = {row.ISBN: row for i, row in books_info.iterrows()}
#pdb.set_trace()

# user feature to list of user_id
print('Preprocessing users....')
country2users = {}
countries = get_unique(users_info.Location)
for country in countries:
    country2users.update({country: users_info['User-ID'][users_info.Location == country].tolist()})
age2users = {}
ages = get_unique(users_info.Age)
for age in ages:
    age2users.update({age: users_info['User-ID'][users_info.Age == age].tolist()})
book2users = {}
for i, row in ratings.iterrows():
    book = row.ISBN
    if book not in book2users:
        book2users.update({book: [row['User-ID']]})
    else:
        book2users[book].append(row['User-ID'])
imp_book2users = {}
for i, row in imp_ratings.iterrows():
    book = row.ISBN
    if book not in imp_book2users:
        imp_book2users.update({book: [row['User-ID']]})
    else:
        imp_book2users[book].append(row['User-ID'])
#pdb.set_trace()

# book feature to list of book_id
print('Preprocessing books....')
author2books = {}
for i, row in books_info.iterrows():
    author = row['Book-Author']
    if author not in author2books:
        author2books.update({author: [row.ISBN]})
    else:
        author2books[author].append(row.ISBN)
publisher2books = {}
for i, row in books_info.iterrows():
    publisher = row.Publisher
    if publisher not in publisher2books:
        publisher2books.update({publisher: [row.ISBN]})
    else:
        publisher2books[publisher].append(row.ISBN)
year2books = {}
for i, row in books_info.iterrows():
    year = row['Year-Of-Publication']
    if year not in year2books:
        year2books.update({year: [row.ISBN]})
    else:
        year2books[year].append(row.ISBN)
user2books = {}
for i, row in ratings.iterrows():
    user = row['User-ID']
    if user not in user2books:
        user2books.update({user: [row.ISBN]})
    else:
        user2books[user].append(row.ISBN)
imp_user2books = {}
for i, row in imp_ratings.iterrows():
    user = row['User-ID']
    if user not in imp_user2books:
        imp_user2books.update({user: [row.ISBN]})
    else:
        imp_user2books[user].append(row.ISBN)
#pdb.set_trace()

# save
print('Saving....')
with open(id2info_path, 'wb') as f:
    pickle.dump((user2info, book2info), f, pickle.HIGHEST_PROTOCOL)
user_dict = {'country': country2users, 'age': age2users, 'reading': book2users, 'imp_reading': imp_book2users}
book_dict = {'author': author2books, 'publisher': publisher2books, 'year': year2books, 'read': user2books, 'imp_read': imp_user2books}
with open(featre2id_path, 'wb') as f:
    pickle.dump((user_dict, book_dict), f, pickle.HIGHEST_PROTOCOL)
