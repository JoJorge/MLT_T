# python 3.6
# preprocessing user data to extract and classification useful info

import numpy as np
import pandas
import pdb

data = []
data_path = '../data/users.csv'
save_path = '../data/users_pre.csv'
num_country_class = 20
age_class_div = [18, 30, 50, 65, float('Inf')]
age_class_name = ['0-18', '19-30', '30-50', '50-65', '65-']

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

# ---------- Classify Age ----------
print('Handling countries....')
print('\tClassifying....')
ages = []
for age in data.Age:
    if np.isnan(age):
        ages.append('unknown')
        continue
    for i in range(len(age_class_div)):
        if age <= age_class_div[i]:
            ages.append(age_class_name[i])
            break
print('\tSaving....')
data.loc[:, 'Age'] = ages            
print('Done\n')

# save CSV
data.to_csv(save_path, index = False)
