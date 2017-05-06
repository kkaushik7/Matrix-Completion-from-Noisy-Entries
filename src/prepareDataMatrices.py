import numpy as np
import sys
import math

import time
import os

from scipy import sparse

def trim(M,E):
	'''
	Perform Matrix Trim as described by Keshavan et. al. (2009)
	'''
	(m,n) = M.shape
	M_Et = M
	# Trim the Columns first
	colSums = np.sum(E,axis=0)
	colMean = np.mean(colSums)
	for col in range(n):
		if sum(E[:,col] >= 2*colMean):
			M_Et = replaceElements(M_Et,0)

	# Trim the Rows next
	rowSums = np.sum(E,axis=1)
	rowMean = np.mean(rowSums)
	for row in range(m):
		if sum(E[row,: ] >= 2*rowMean):
			M_Et = replaceElements(M_Et,0)
	return M_Et

def adjust_ids(ids):
	sorted_indices = ids.argsort()

	sorted_ids = ids[sorted_indices]

	unique_ids = np.unique(sorted_ids)

	sorted_ids_adjusted = np.zeros(sorted_ids.size,np.uint32)

	curr_id = sorted_ids[0]
	curr_index = 0
	for i in xrange(sorted_ids.size):
		if sorted_ids[i]==curr_id:
			sorted_ids_adjusted[i]=curr_index
		else:
			curr_id=sorted_ids[i]
			curr_index+=1
			sorted_ids_adjusted[i]=curr_index
		print i

	return (sorted_indices, sorted_ids_adjusted)

no_of_movies = 2000

training_directory = 'netflix/training_set/'
list_of_movies = os.listdir(training_directory)
list_of_movies.sort()

no_of_ratings = np.zeros((len(list_of_movies)))

movie_count = 0
for movie in list_of_movies:

	ratings_for_movie = sum(1 for line in open(training_directory+movie)) - 1
	no_of_ratings[movie_count] = ratings_for_movie

	movie_count += 1
	print movie_count

most_ratings_indices = np.argsort(-no_of_ratings)
list_of_movies = np.array(list_of_movies)
list_of_movies = list_of_movies[most_ratings_indices]
most_ratings_indices = most_ratings_indices[0:no_of_movies]
list_of_movies = list_of_movies[0:no_of_movies]

total_ratings = np.sum(no_of_ratings[most_ratings_indices])


rating_row = np.zeros((int(total_ratings)),np.uint32)
rating_col = np.zeros((int(total_ratings)),np.uint32)
rating_data = np.zeros((int(total_ratings)))

curr_rating = 0
curr_movie = 0

for movie in list_of_movies:

	curr_movie += 1
	print(curr_movie)

	f = open(training_directory+movie)

	first_line = 1
	for line in f.readlines():
		if(first_line==1):
			row = int(line.strip()[:-1]) - 1 # get the movie ID for this file
			first_line = 0
		else:
			line = line.split(',')
			col = int(line[0]) - 1 # get the user ID for this rating
			data = float(line[1]) # get the actual rating

			if(col<=150000): # 5645 (999), 55393 (9999), 549880 (99999), 98133 (17769)
				rating_row[curr_rating] = row
				rating_col[curr_rating] = col
				rating_data[curr_rating] = data

				curr_rating += 1

	f.close()

rating_col = rating_col[0:curr_rating]
rating_row = rating_row[0:curr_rating]
rating_data = rating_data[0:curr_rating]

# sort by row first to get unique movie IDs
(sorted_indices,rating_row) = adjust_ids(rating_row)
rating_col = rating_col[sorted_indices]
rating_data = rating_data[sorted_indices]

(sorted_indices,rating_col) = adjust_ids(rating_col)
rating_row = rating_row[sorted_indices]
rating_data = rating_data[sorted_indices]

M0 = sparse.coo_matrix((rating_data, (rating_row, rating_col)), shape=(np.max(rating_row)+1, np.max(rating_col)+1))

M0 = M0.toarray()

#remove ~50% of the ratings for testing
m,n = M0.shape
E_train = 1 - np.ceil(np.random.rand(m,n) - 0.50)

M_E = np.multiply(M0,E_train)


np.save('data_matrices/M0.npy',M0)
np.save('data_matrices/M_E.npy',M_E)
