from typing import List, Dict
import knn_kv as knn
import sys
from random import random, randint

r: float
train_dataset: List[dict] = [
    {'features': [73.96, 35.91], 'result': 0.0},
    {'features': [94.52, 27.7], 'result': 163.22531050462305},
    {'features': [96.84, 37.33], 'result': 235.63927544343713},
    {'features': [78.63, 23.62], 'result': 142.88713260897111},
    {'features': [70.48, 4.12], 'result': 47.908428185855705},
    {'features': [53.05, 43.4], 'result': 0.0},
    {'features': [51.71, 5.69], 'result': 23.342692416635074},
    {'features': [78.83, 47.76], 'result': 0.0},
    {'features': [96.94, 40.7], 'result': 239.0906054126505},
    {'features': [58.31, 14.05], 'result': 0.0},
    {'features': [91.54, 12.77], 'result': 73.97575564360035},
    {'features': [82.96, 12.45], 'result': 91.70105627116332},
    {'features': [78.4, 1.25], 'result': 13.356611903963218},
    {'features': [82.19, 39.08], 'result': 0.0},
    {'features': [89.26, 25.76], 'result': 168.28939975417453},
    {'features': [55.67, 23.73], 'result': 0.0},
    {'features': [97.05, 12.35], 'result': 78.77340773822712},
    {'features': [88.68, 17.16], 'result': 110.91466182684353},
    {'features': [53.52, 30.58], 'result': 0.0},
    {'features': [93.92, 2.17], 'result': 16.425293468800678},
    {'features': [54.32, 30.87], 'result': 0.0},
    {'features': [68.97, 31.4], 'result': 0.0},
    {'features': [96.46, 21.4], 'result': 102.74141323532731},
    {'features': [97.1, 9.94], 'result': 56.17837680578927},
    {'features': [80.24, 33.86], 'result': 63.86849678774686},
    {'features': [70.92, 23.34], 'result': 104.86886187999121},
    {'features': [97.72, 11.68], 'result': 72.97062330443988},
    {'features': [93.59, 21.9], 'result': 140.7021644108643},
    {'features': [61.78, 4.93], 'result': 62.19944523862733},
    {'features': [87.6, 34.3], 'result': 211.44496057763516},
    {'features': [76.49, 26.02], 'result': 178.13658096402128},
    {'features': [97.44, 28.93], 'result': 133.38436990432663},
    {'features': [53.04, 39.65], 'result': 0.0},
    {'features': [69.4, 17.87], 'result': 159.06828186845195},
    {'features': [95.83, 0.17], 'result': 5.38872134698453},
    {'features': [82.32, 9.61], 'result': 64.13984201361943},
    {'features': [70.81, 7.75], 'result': 72.59075048775946},
    {'features': [96.18, 48.52], 'result': 147.5276029424029},
    {'features': [80.87, 40.76], 'result': 0.0},
    {'features': [87.74, 43.32], 'result': 0.0},
    {'features': [51.32, 27.54], 'result': 0.0},
    {'features': [76.06, 11.97], 'result': 81.16333960592738},
    {'features': [64.67, 26.92], 'result': 0.0},
    {'features': [63.63, 43.06], 'result': 0.0},
    {'features': [56.86, 34.71], 'result': 0.0},
    {'features': [82.12, 34.31], 'result': 108.13951031951315},
    {'features': [54.72, 4.76], 'result': 124.67994659173367},
    {'features': [63.02, 32.99], 'result': 0.0},
    {'features': [73.73, 23.27], 'result': 212.4492949652139},
]

test_sample: dict = {'features': [57.95, 4.97], 'result': 90.26847751986058}
print('real result is: %f' % round(test_sample['result'], 2))

# prediction result with k-nn
print('k-nn')
r = knn.knnestimate(train_dataset, test_sample['features'], k=5)
r = round(r, 2)
print('k-nn(5) result: %f' % r)

r = knn.knnestimate(train_dataset, test_sample['features'], k=4)
r = round(r, 2)
print('k-nn(4) result: %f' % r)

r = knn.knnestimate(train_dataset, test_sample['features'])
r = round(r, 2)
print('k-nn(3) result: %f' % r)

# weightedknn
print('weighted k-nn')
r = knn.weightedknn(
    train_dataset,
    test_sample['features'],
    k=5,
    weightf=knn.gaussian)
r = round(r, 2)
print('weighted k-nn(5),gaussian result: %f' % r)

r = knn.weightedknn(
    train_dataset,
    test_sample['features'],
    k=5,
    weightf=knn.inverseweight)
r = round(r, 2)
print('weighted k-nn(5),inverseweight result: %f' % r)

r = knn.weightedknn(
    train_dataset,
    test_sample['features'],
    k=5,
    weightf=knn.subtractweight)
r = round(r, 2)
print('weighted k-nn(5),subtractweight result: %f' % r)
