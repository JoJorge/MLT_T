# python 3.6
# preprocessing user data to extract and classification useful info

import numpy as np
import pandas
import pdb

data = []
data_path = '../data/users.csv'
save_path = '../data/users_DT.csv'
num_country_class = 20

data = pandas.read_csv(data_path)

print(data.keys(), '\n')
# pdb.set_trace()

# ---------- Classify Countries ----------
print('Handling countries....')
# get locations and leave country only
print('\tSpliting....')
countries = []
for loc in data['Location']:
    countries.append(loc.split(',')[-1].strip())

# get top N - 1 country except empty
print('\tClassifying....')
cnt_country = {}
for country in countries:
    if country in cnt_country:
        cnt_country[country] += 1
    else:
        cnt_country.update({country: 1})
sorted_cnt_country = sorted(cnt_country.items(), key = lambda x:x[1], reverse = True) 
country_class_name = []
for item in sorted_cnt_country:
    country = item[0]
    if country != '':
        country_class_name.append(country)
    if len(country_class_name) == num_country_class - 1:
        break
for i in range(len(countries)):
    if countries[i] not in country_class_name:
        countries[i] = 'others'
print('\tSaving....')
data.loc[:, 'Location'] = countries
country_class_name.append('others')
print('Done\n')

# ---------- Age ----------
# change nan to 0
print('Handling Age....')
ages = []
for age in data.Age:
    if np.isnan(age):
        ages.append(0)
    else:
        ages.append(age)
print('\tSaving....')
data.loc[:, 'Age'] = ages            
print('Done\n')

# save CSV
data.to_csv(save_path, index = False)
