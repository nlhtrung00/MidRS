import os
import pathlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import pandas as pd
from scipy import sparse
from algorithms.constants import *
# from constants import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score
import math
from sklearn.model_selection import train_test_split



class CollaborativeFiltering():
    def __init__(self, Y_data, Y_test, k=2, amount=5, similarity_based=COSINE, based=USER):
        self.Y_data = Y_data.astype(np.float32) if based == USER else Y_data[:, [
            1, 0, 2]].astype(np.float32)
        self.Y_test = Y_test 
        self.n_users_test = int(np.max(self.Y_test[:, 0])) + 1 if based == USER else int(np.max(self.Y_test[:, 1])) + 1
        self.n_items_test = int(np.max(self.Y_test[:, 1])) + 1 if based == USER else int(np.max(self.Y_test[:, 0])) + 1
        self.k = k
        self.amount = amount
        self.similarity_based = similarity_based
        self.based = based
        self.Y_bar = None
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 if self.n_users_test<int(np.max(self.Y_data[:, 0])) + 1 else self.n_users_test
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1 if self.n_items_test<int(np.max(self.Y_data[:, 1])) + 1 else self.n_items_test
        self.Similarity = []

    def normalized(self):
        users = self.Y_data[:, 0]
        self.Y_bar = self.Y_data.copy()
        self.hat = np.zeros(self.n_users)
        if self.similarity_based != COSINE and self.similarity_based != JACCARD:
            
            for user in range(self.n_users):
                index_user_rated = np.where(users == user)[0]
                ratings = self.Y_data[index_user_rated, 2]
                if len(ratings) == 0: average_user_rated = 0
                else: average_user_rated = np.mean(ratings)
                if np.isnan(average_user_rated):
                    average_user_rated = 0
                self.hat[user] = average_user_rated
                self.Y_bar[index_user_rated, 2] = ratings - self.hat[user]
        self.Y_normalized = sparse.coo_matrix(
            (self.Y_bar[:, 2], (self.Y_bar[:, 0], self.Y_bar[:, 1])), (self.n_users, self.n_items)).tocsr()


    def similarity(self):
        # self.Similarity = cosine_similarity(
        #     self.Y_normalized, self.Y_normalized) if self.similarity_based != JACCARD else 1 - pairwise_distances(self.Y_normalized, metric="hamming")
        if self.similarity_based != JACCARD:
            self.Similarity = cosine_similarity(self.Y_normalized, self.Y_normalized) 
        else:
            matrix_array = []
            for i in range(self.n_users):
                lst_temp = []
                for j in range(self.n_users):
                    user1, user2 = self.convert(i,j)
                    lst_temp.append(jaccard_similarity(user1,user2))
                matrix_array.append(lst_temp)
            self.Similarity = np.asarray(matrix_array)
# arr = numpy.array(lst)
#             if self.similarity_based != JACCARD else 1 - pairwise_distances(self.Y_normalized.T,metric="hamming")

    def fit(self):
        self.normalized()
        self.similarity()

    def __pred(self, u, i):
        # normalized = 1 third param if want to plus self.hat[user]
        index_users_rated_i = np.where(self.Y_data[:, 1] == i)[0]
        users_rated_i = (self.Y_data[index_users_rated_i, 0]).astype(np.int32)
        sim = self.Similarity[u, users_rated_i]
        index_knn = np.argsort(sim)[-self.k:]
        nearest_knn = sim[index_knn]
        ratings_users_item = self.Y_normalized[users_rated_i[index_knn], i].T
        return (ratings_users_item*nearest_knn)[0]/(np.abs(nearest_knn).sum() + 1e-8) + self.hat[u]

    def pred(self, u, i):
        u = int(u)
        i = int(i)
        if self.based == USER: return self.__pred(u, i)
        return self.__pred(i, u)

    def recommend(self, u):
        index_user_rated = np.where(self.Y_data[:, 0] == u)[0]
        index_user_rated_test = np.where(self.Y_test[:, 0] == u)[0]
        items_rated_by_user = self.Y_data[index_user_rated, 1].astype(np.int32)
        items_rated_by_user_test = self.Y_test[index_user_rated_test, 1].astype(np.int32)
        items_rated_by_user_all = items_rated_by_user.tolist() + items_rated_by_user_test.tolist()
        recommended_items = []
        for i in range(self.n_items):
            if i not in items_rated_by_user_all:
                rating = self.__pred(u, i)
                recommended_items.append([i,rating])
        temp = np.array(recommended_items)
        index = np.argsort(temp[:, 1])[-self.amount:]
        return temp[index,0].astype(np.int32)[::-1]

    def recommendation_result(self):
        result = []
        for u in range(self.n_users):
            result.append([])
            result[u].append(u)
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            row = recommended_items.tolist()
            result[u] += row
        if self.based == USER:
            return result
        convert = []
        for i in range(self.n_items):
            convert.append([])
            convert[i].append(i)
        for i in range(len(result)):
            item = result[i][0]
            users = result[i][1:]
            for user in range(len(users)):
                convert[users[user]].append(item)
        return convert

    def evaluate(self):
        actual = self.Y_test[:, 2]
        predicted = []
        n_tests = self.Y_test.shape[0]
        if self.similarity_based == JACCARD:
            for n in range(n_tests):
                pred = self.pred(self.Y_test[n, 0], self.Y_test[n, 1])
                if pred >= 0.5:
                    predicted.append(1)
                else:
                    predicted.append(0)
            precision = precision_score(actual, predicted)
            recall = recall_score(actual, predicted)
            return precision, recall

        for n in range(n_tests):
                predicted.append(self.pred(self.Y_test[n, 0], self.Y_test[n, 1]))
        MSE = mean_squared_error(actual, predicted)
        RMSE = math.sqrt(MSE)

        MAE = mean_absolute_error(actual, predicted)
        return RMSE, MAE

    def convert(self, u1, u2):
        item_list = list(range(0, self.n_items))
        users = self.Y_data[:, 0]
        index_user1_rated = np.where(users == u1)[0]
        index_user2_rated = np.where(users == u2)[0]
        items_user1_rated = self.Y_data[index_user1_rated, 1]
        items_user2_rated = self.Y_data[index_user2_rated, 1]
        ratings_user1 = self.Y_data[index_user1_rated, 2].tolist()
        ratings_user2 = self.Y_data[index_user2_rated, 2].tolist()
        out_item1 = list(set(item_list)-set(items_user1_rated))
        out_item2 = list(set(item_list)-set(items_user2_rated))
        for i in range(len(out_item1)):
            ratings_user1.insert(out_item1[i],-1)
        for i in range(len(out_item2)):
            ratings_user2.insert(out_item2[i],-1)
        i = 0
        while i<len(ratings_user1):
            if ratings_user1[i] == -1 or ratings_user2[i] == -1:
                del ratings_user1[i]
                del ratings_user2[i]
            else:
                i += 1
        return ratings_user1, ratings_user2

def jaccard_similarity(list1, list2):
    intersection = np.logical_and(list1, list2)
    union = np.logical_or(list1, list2)
    if union.sum() == 0: 
        return 0
    similarity = intersection.sum() / float(union.sum())
    return similarity

# def get_data(path):
#     ratings = pd.read_csv(path, sep = "[;, \t]", header=None, encoding='latin-1', engine='python')
#     Y_data = ratings.to_numpy()
#     return Y_data


# parent_path = pathlib.Path(__file__).parent.parent.resolve()
# path = os.path.join(parent_path,'uploads','2022-11-27_172505.523213-data-r.csv')
# x = get_data(path)

# parent_path = pathlib.Path(__file__).parent.resolve()
# path = os.path.join(parent_path,'data-150-binary.csv')
# x = get_data(path)

# for i in range(x.shape[0]-1):
#     for j in range(i+1,x.shape[0]-1):
#         if x[i][0] == x[j][0] and x[i][1] == x[j][1] :
#             print(i,j)
# x[:, :2] -= 1
# print(x.shape[0])
# Ydata, Ytest = train_test_split(x)
# print(Ydata)
# print(Ytest)
# rs_cf_cosine_user = CollaborativeFiltering(Y_data = Ydata, Y_test= Ytest, k=30, amount = 5, similarity_based=JACCARD, based=USER)
# print(rs_cf_cosine_user.n_users, rs_cf_cosine_user.n_items)
# rs_cf_cosine_user.normalized()
# rs_cf_cosine_user.similarity()
# rs_cf_cosine_user.fit()
# print(rs_cf_cosine_user.Y_normalized)
# print(rs_cf_cosine_user.Similarity)
# for i in rs_cf_cosine_user.Similarity:
#     print(i)
# print(rs_cf_cosine_item.n_users, rs_cf_cosine_item.n_items)

# rs_cf_cosine_item.fit()
# print(rs_cf_cosine_user.recommendation_result())
# print(rs_cf_cosine_user.evaluate())

# eva_cosine_item = rs_cf_cosine_item.evaluate()
# print(eva_cosine_item)
# rs = CollaborativeFiltering(Y_data=Ydata, Y_test=Ytest, k=2, amount=10, similarity_based=PEARSON,based=USER)
# # # a, b = rs.convert(0,1)
# # # print(jaccard_similarity(a,b))
# rs.normalized()
# rs.similarity()
# print(rs.Y_bar)
# print(rs.Y_normalized)
# print(rs.Similarity)
# print(rs.evaluate())
# # print(rs.recommendation_result())
# # print(rs.Similarity)

