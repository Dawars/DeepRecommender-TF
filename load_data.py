import csv
import array

users = array(2000)

print(users)
with open('./mangaki-data-challenge-0908/watched.csv', newline='') as csvfile:
    users_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(users_reader, None)  # skip header

    for row in users_reader:
        user_id, work_id, rating = row
        print(user_id)
        users[int(user_id)].append({'work_id': work_id, 'rating': rating})

print(users[:100])