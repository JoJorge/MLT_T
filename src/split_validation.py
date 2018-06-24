# python 3.6
# split validation data w/ and wo/ rating

import csv
import numpy as np

percent_valid_w_rating = 5
percent_valid_wo_rating_user = 5
percent_valid_wo_rating_book = 5
max_split_rating_per_ub = 10
keys = []
data = []
data_path = '../data/'

def split_wo_data(IDs, rating_per_ID, need_percent): 
    final_n, total_num = 1, 0
    valid = []
    for n in range(1, max_split_rating_per_ub+1):
        for i in range(lenData):
            ID = IDs[i]
            if rating_per_ID[ID] == n:
                valid.append(data[i])
                used_data[i] += 1
                total_num += 1
                if total_num / lenData * 100 >= need_percent:
                    final_n = n
                    return final_n, valid
        percent = total_num / lenData * 100
        print('Percent of pair for user with rating less than and equal to %d: %.2f%%'%(n, percent))
    return max_split_rating_per_ub, valid
def write_rating(file_name, writing_data):
    with open(data_path + file_name, 'w', newline = '', encoding = 'utf-8') as csvf:
        writer = csv.writer(csvf, delimiter = ',', quotechar = '"')
        writer.writerow(keys)
        for rating in writing_data:
            writer.writerow(rating)
    
# read
with open(data_path + 'book_ratings_train.csv', newline = '', encoding = 'utf-8') as csvf:
    reader = csv.reader(csvf, delimiter = ',', quotechar = '"')
    
    for line in reader:
        data.append(line)
    keys = data[0]
    data = data[1:]

lenData = len(data)
print(keys)
print('Number of pairs:', lenData, '\n')

print('Size of validation w/ rating: %d%%'%(percent_valid_w_rating))
print('Size of validation w.o/ rating for user: %d%%'%(percent_valid_wo_rating_user))
print('Size of validation w.o/ rating for book: %d%%\n'%(percent_valid_wo_rating_book))

# get number of rating per user/book
rating_per_user = {}
rating_per_book = {}
for pair in data:
    if pair[0] in rating_per_user:
        rating_per_user[pair[0]] += 1
    else:
        rating_per_user.update({pair[0]: 1})
    if pair[1] in rating_per_book:
        rating_per_book[pair[1]] += 1
    else:
        rating_per_book.update({pair[1]: 1})
print('Total users:', len(rating_per_user))
print('Total books:', len(rating_per_book), '\n')
        
# get sorted rating per user/book        
rpu_vlst = list(rating_per_user.values())
sorted_idx_rpu = sorted(range(len(rpu_vlst)), key = lambda i: rpu_vlst[i])
rpb_vlst = list(rating_per_book.values())
sorted_idx_rpb = sorted(range(len(rpb_vlst)), key = lambda i: rpb_vlst[i])

used_data = np.zeros((lenData))
# get appropriate n for split validation data w.o/ rating for user and get user validation data
final_n_user, user_valid = split_wo_data([x[0] for x in data], rating_per_user, percent_valid_wo_rating_user)
print('Final selected n for user:', final_n_user)
print('Final validation size for user: %.2f%%\n'%(len(user_valid) / lenData * 100))

# get appropriate n for split validation data w.o/ rating for user and get user validation data
final_n_book, book_valid = split_wo_data([x[1] for x in data], rating_per_book, percent_valid_wo_rating_book)
print('Final selected n for book:', final_n_book)
print('Final validation size for book: %.2f%%\n'%(len(book_valid) / lenData * 100))

# print some analysis
overlap = np.sum(used_data == 2)
for i in range(lenData):
    if used_data[i] > 0:
        rating_per_user[data[i][0]] -= 1
        rating_per_book[data[i][1]] -= 1
cnt_unknown_book_in_user, cnt_unknown_user_in_book = 0, 0
for rating in user_valid:
    if rating_per_book[rating[1]] == 0:
        cnt_unknown_book_in_user += 1
for rating in book_valid:
    if rating_per_user[rating[0]] == 0:
        cnt_unknown_user_in_book += 1
print('Overlap: %.2f%%\n'%(overlap / lenData * 100))
print('Unrated books in user_valid: %.2f%%/%.2f%%'%(cnt_unknown_book_in_user/lenData*100, len(user_valid)/lenData*100))
print('Unrating users in book_valid: %.2f%%/%.2f%%\n'%(cnt_unknown_user_in_book/lenData*100, len(book_valid)/lenData*100))

# write validation for user and book
write_rating('user_valid.csv', user_valid)
write_rating('book_valid.csv', book_valid)

# split validation with rating (make sure every user and book has rating)
total_num = 0
rating_valid = []
for i in range(lenData):
    if used_data[i] > 0:
        continue
    used_data[i] = 1
    rating = data[i]
    if rating_per_user[rating[0]] <= 1 or rating_per_book[rating[1]] <= 1:
        continue
    total_num += 1
    rating_valid.append(rating)
    rating_per_user[rating[0]] -= 1
    rating_per_book[rating[1]] -= 1
    if total_num / lenData * 100 >= percent_valid_w_rating:
        break
print('Final validation size for normal rating: %.2f%%\n'%(len(rating_valid) / lenData * 100))

# write validation data
write_rating('valid.csv', rating_valid)

# write split train data
data_split = []
for i in range(lenData):
    if used_data[i] == 0:
        data_split.append(data[i])
write_rating('train_split.csv', data_split)
