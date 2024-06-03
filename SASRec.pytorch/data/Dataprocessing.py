import pickle
from collections import defaultdict

'''
Loading Data:
    Loads data from the reviews.pickle file.
Processing Data:
    Counts the occurrences of each user and item.
    Writes the user-item interactions to a text file (reviews_dataset_name.txt).
    Filters out users and items with fewer than 5 interactions.
    Maps users and items to unique numerical IDs.
    Sorts reviews by rating within each user.
Saving Processed Data:
    Writes the processed user-item interactions to a text file (dataset_name.txt).
'''

# Load data from reviews.pickle
with open('/home/hajar.laktaoui/ImplementationFolder/TripAdvisor/reviews.pickle', 'rb') as f:

    data = pickle.load(f)

countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

dataset_name = 'TripAdvisor'
f = open('reviews_' + dataset_name + '.txt', 'w')

for l in data:
    line += 1
    f.write(" ".join([l['user'], l['item']]) + '\n')
    rev = l['user']
    asin = l['item']
    countU[rev] += 1
    countP[asin] += 1

f.close()

usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
User = dict()

for l in data:
    asin = l['item']
    rev = l['user']
    if countU[rev] < 5 or countP[asin] < 5:
        continue

    if rev in usermap:
        userid = usermap[rev]
    else:
        usernum += 1
        userid = usernum
        usermap[rev] = userid
        User[userid] = []
    if asin in itemmap:
        itemid = itemmap[asin]
    else:
        itemnum += 1
        itemid = itemnum
        itemmap[asin] = itemid
    User[userid].append([l['rating'], itemid])

# Sort reviews in User according to rating (assuming rating is equivalent to time)

for userid in User.keys():
    User[userid].sort(key=lambda x: x[0])

print(usernum, itemnum)

f = open('/home/hajar.laktaoui/ImplementationFolder/SASRec.pytorch/data/TripAdvisor.txt', 'w')

for user in User.keys():
    for i in User[user]:
        f.write('%d %d\n' % (user, i[1]))

f.close()
