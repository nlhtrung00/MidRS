import os
import pathlib
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
from scipy import sparse
# from scipy.stats import pearsonr
import constants


class CollaborativeFiltering():
    def __init__(self, Y_data, k=2, similarity_based=constants.JACCARD, based=constants.USER):
        self.Y_data = Y_data if based == constants.USER else Y_data[:, [
            1, 0, 2]]
        self.k = k
        self.similarity_based = similarity_based
        self.Y_bar = None
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def normalized(self):
        users = self.Y_data[:, 0]
        self.Y_bar = self.Y_data.copy()
        self.hat = np.zeros(self.n_users)
        if self.similarity_based != constants.CONSINE and self.similarity_based != constants.JACCARD:
            for user in range(self.n_users):
                index_user_rated = np.where(users == user)[0]
                ratings = self.Y_data[index_user_rated, 2]
                average_user_rated = np.mean(ratings)
                if np.isnan(average_user_rated):
                    average_user_rated = 0
                self.hat[user] = average_user_rated
                self.Y_bar[index_user_rated, 2] = ratings - self.hat[user]

        self.Y_normalized = sparse.coo_matrix(
            (self.Y_bar[:, 2], (self.Y_bar[:, 0], self.Y_bar[:, 1])), (self.n_users, self.n_items)).tocsr()
        print(self.Y_normalized)

    def similarity(self):
        self.Similarity = 1 - pairwise_distances(
            self.Y_normalized.todense(), metric="hamming")
        print(self.Similarity)

    # def __pred(self, u, i):
    #     # normalized = 1 third param if want to plus self.hat[user]
    #     index_users_rated_i = np.where(self.Y_data[:, 1] == i)[0]
    #     users_rated_i = (self.Y_data[index_users_rated_i, 0]).astype(np.int32)
    #     sim = self.Similarity[u, users_rated_i]
    #     index_knn = np.argsort(sim)[-self.k:]
    #     nearest_knn = sim[index_knn]
    #     ratings_users_item = self.Y_normalized[users_rated_i[index_knn], i].T
    #     return (ratings_users_item*nearest_knn)[0]/(np.abs(nearest_knn).sum() + 1e-8)

    # def recommend(self, u):
    #     index_user_rated = np.where(self.Y_data[:, 0] == u)[0]
    #     items_rated_by_user = self.Y_data[index_user_rated, 1].astype(np.int32)
    #     recommended_items = []
    #     for i in range(self.n_items):
    #         if i not in items_rated_by_user:
    #             rating = self.__pred(u, i)
    #             if rating > 0:
    #                 recommended_items.append(i)
    #     return recommended_items

    # def print_recommendation(self):
    #     self.normalized()
    #     self.similarity()
    #     result = []
    #     for u in range(self.n_users):
    #         recommended_items = self.recommend(u)
    #         row = [u] + recommended_items
    #         result.append(row)
    #     return result
    #         # print('Recommend item(s):', recommended_items, 'to user', u)


def get_data(path):
    ratings = pd.read_csv(path, sep=' ', header=None, encoding='latin-1')
    Y_data = ratings.to_numpy()
    return Y_data


parent_path = pathlib.Path(__file__).parent.resolve()
path = os.path.join(parent_path, 'data copy.csv')
x = get_data(path)
rs = CollaborativeFiltering(x)
rs.normalized()
rs.similarity()
# rs = Popular(x)
# print(rs.recommendation_result())
# print(rs.print_recommendation())

# rs = CollaborativeFiltering(x, k=2, similarity_based=constants.CONSINE)
# print(rs.print_recommendation())
# users = x[:,0]
# mu = np.zeros((7,))
# y = x
# for n in range(7):
#             # row indices of rating done by user n
#             # since indices need to be integers, we need to convert
#     index_user_rated = np.where(users == n)[0]
#     item_ids = x[index_user_rated, 1]
#     ratings = x[index_user_rated, 2]
#     m = np.mean(ratings)
#     mu[n] = m
#     y[index_user_rated,2] = ratings - mu[n]
#     normalized = sparse.coo_matrix((y[:, 2],(y[:, 0], y[:, 1])),(7, 5)).tocsr()

# s = cosine_similarity(normalized, normalized)
# print(s)
# index_users_rated_i = np.where(x[:, 1] == 1)[0]
# users_rated_i = x[index_users_rated_i, 0].astype(np.int32)
# sim = s[1, users_rated_i]

# knn = np.argsort(sim)[-2:]
# nearest_s = sim[knn]


# print(users_rated_i[knn])
# r = normalized[users_rated_i[knn],1]
# # => why T
# print(r)

# rT = normalized[users_rated_i[knn],1].T
# # => why T
# print(rT)

# print(nearest_s)

# print(r*nearest_s)
# # r_temp = (r*nearest_s)[0]/((np.abs(nearest_s).sum() + 1e-8))
# # print(r_temp)


# # index_user_rated = np.where(x[:, 0] == 1)[0]
# # print(index_user_rated)
# # items_rated_by_u = x[index_user_rated, 1].astype(np.int32)
# # print(items_rated_by_u)

# # .tolist()
# recommended_items = []
# # /(np.abs(nearest_knn).sum() + 1e-8)

# # users = self.Y_data[:,0]
# #         self.Y_bar = self.Y_data.copy()
# #         self.hat = np.zeros((self.n_users,))
