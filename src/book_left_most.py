import pdb
import pandas
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--left-authors', type=int, default=100)
parser.add_argument('--left-publishers', type=int, default=100)
parser.add_argument('--outfile', type=str, default='../data/book_left_most.csv')
args = parser.parse_args()

book_table = pandas.read_csv("../data/books.csv", usecols=['ISBN','Book-Author','Year-Of-Publication','Publisher'])
book_num = book_table.shape[0]
def left_most(data_table, col, left_num):
    count_dict = {}
    for x in data_table[col]:
        if x in count_dict:
            count_dict[x] += 1
        else:
            count_dict[x] = 1
    left_list = sorted(count_dict.items(), key = lambda x:x[1], reverse = True) 
    left_classes = list(map(lambda x: x[0], left_list[:left_num]))
    items = [ x if x in left_classes else "others" for x in data_table[col] ]
    data_table.loc[:, col] = items
    #pdb.set_trace()

left_most(book_table, 'Book-Author', args.left_authors)
left_most(book_table, 'Publisher', args.left_publishers)
book_table.to_csv("../data/books_pre_most.csv", index = False)
