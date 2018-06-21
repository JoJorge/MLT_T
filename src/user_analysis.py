# python 3.6
# get and analysis users' country

import csv

# read
keys = []
data = []
data_path = '../data/users.csv'
with open(data_path, newline = '', encoding = 'utf-8') as csvf:
    reader = csv.reader(csvf, delimiter = ',', quotechar = '"')
    
    for line in reader:
        data.append(line)
    keys = data[0]
    data = data[1:]

print(keys, '\n')

# get locations
locations = []
for line in data:
    line = line[1].split(',')
    locations.append([word.strip() for word in line])

# analysis countries
countries = [loc[-1] for loc in locations]
cnt_country = {}
for country in countries:
    if country in cnt_country:
        cnt_country[country] += 1
    else:
        cnt_country.update({country: 1})
sorted_cnt_country = sorted(cnt_country.items(), key = lambda x:x[1], reverse = True)    

# print result
print('Number of users:', len(data))
print('Number of country:', len(cnt_country.keys()))
print('Top 20 countries: ')
top20 = 0
for i in range(20):
    print('\t' + sorted_cnt_country[i][0] + ':', sorted_cnt_country[i][1])
    top20 += sorted_cnt_country[i][1]
print('Total user number of top 20:', top20)

    