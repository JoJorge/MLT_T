#import pdb
import pandas

book_table = pandas.read_csv("../data/books.csv", usecols=['ISBN','Book-Author','Year-Of-Publication','Publisher'])
book_num = book_table.shape[0]
#t.iloc[]
#year_table = book_table.sort_values(by='Year-Of-Publication', ascending=True)

period_class_bound = [1800, 1950, 1960, 1970, 1980, 1990, 2000, float('Inf')]
period_class = ["<1800", "<1950", "<1960", "<1970", "<1980", "<1990", "<2000", ">=2001"]
table = book_table['Year-Of-Publication']
years = []
for i in range(book_num):
    year = table[i]
    for cls, upper_bound in enumerate(period_class_bound):
        if year < upper_bound:
            years.append(period_class[cls])
            break
book_table.loc[:, 'Year-Of-Publication'] = years

def elim_one(data_table, col):
    count_dict = {}
    for x in data_table[col]:
        if x in count_dict:
            count_dict[x] += 1
        else:
            count_dict[x] = 1
    items = [ "only_1" if count_dict[x] == 1 or type(x)==float else x for x in data_table[col] ]
    data_table.loc[:, col] = items
    #pdb.set_trace()
    #count_list = sorted(count_dict.items(), key = lambda x:x[1], reverse = True) 

elim_one(book_table, 'Book-Author')
elim_one(book_table, 'Publisher')
book_table.to_csv("../data/books_pre.csv", index = False)

#pdb.set_trace()
#year_table.iloc[0].name

