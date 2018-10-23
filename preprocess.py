


filename = './yelp_academic_dataset_review.json'
with open(filename, 'r') as f:
    review_list = []
    for review in f:
        review_list.append(review)
print(len(review_list))