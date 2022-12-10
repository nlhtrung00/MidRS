import math
import os
import pathlib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from algorithms.constants import *
# from constants import *
from sklearn.model_selection import train_test_split

class MatrixFactorizattion():
    def __init__(self, Y_data, Y_test, amount=5, latent=10, regularization=0.1, Xinit=None, Winit=None, eta=0.5, max_iterations=100):
        self.Y_raw_data = Y_data.astype(np.float32)
        self.Y_test = Y_test.astype(np.float32)
        self.n_users_test = int(np.max(self.Y_test[:, 0])) + 1 
        self.n_items_test = int(np.max(self.Y_test[:, 1])) + 1 
        self.amount = amount
        self.latent = latent
        self.regularization = regularization
        self.eta = eta
        self.max_iterations = max_iterations
        # self.based = based

        self.n_users = int(np.max(self.Y_raw_data[:, 0])) + 1 if self.n_users_test<int(np.max(self.Y_raw_data[:, 0])) + 1 else self.n_users_test
        self.n_items = int(np.max(self.Y_raw_data[:, 1])) + 1 if self.n_items_test<int(np.max(self.Y_raw_data[:, 1])) + 1 else self.n_items_test

        self.n_ratings = Y_data.shape[0]

        if Xinit is None:
            self.X = np.random.randn(self.n_items, latent)
        else:
            self.X = Xinit

        if Winit is None:
            self.W = np.random.randn(latent, self.n_users)
        else:
            self.W = Winit
        
    def normalize(self):
        # if self.based == USER:
        # user_col = 0
        # n_objects = self.n_users
        # else:
        #     user_col = 1
        #     n_objects = self.n_items

        # users = self.Y_raw_data[:, user_col]
        self.Y_bar = self.Y_raw_data.copy()
        # self.hat = np.zeros(n_objects,)
        # for user in range(n_objects):
        #     index_user_rated = np.where(users == user)[0]
        #     ratings = self.Y_raw_data[index_user_rated, 2]
        #     if len(ratings) == 0: average_user_rated = 0
        #     else: average_user_rated = np.mean(ratings)
        #     if np.isnan(average_user_rated):
        #         average_user_rated = 0
        #     self.hat[user] = average_user_rated
        #     self.Y_bar[index_user_rated, 2] = ratings - self.hat[user]

    # def loss(self):
    #     L = 0
    #     for i in range(self.n_ratings):
    #         user, item, rating = int(self.Y_bar[i, 0]), int(
    #             self.Y_bar[i, 1]), int(self.Y_bar[i, 2])
    #         L += 0.5*(rating - self.X[item, :].dot(self.W[:, user]))**2
    #     L /= self.n_ratings
    #     L += 0.5*self.regularization * \
    #         (np.linalg.norm(self.X, 'fro')+np.linalg.norm(self.W, 'fro'))
    #     return L

    def get_items_rated_by_user(self, user_id):
        index_items_rated_by_user = np.where(self.Y_bar[:, 0] == user_id)[0]
        items_rated = self.Y_bar[index_items_rated_by_user, 1].astype(np.int32)
        ratings = self.Y_bar[index_items_rated_by_user, 2]
        return (items_rated, ratings)

    def get_users_who_rate_item(self, item_id):
        index_users_rating_item = np.where(self.Y_bar[:, 1] == item_id)[0]
        users_rating = self.Y_bar[index_users_rating_item, 0].astype(np.int32)
        ratings = self.Y_bar[index_users_rating_item, 2]
        return (users_rating, ratings)

    def updateX(self):
        for item in range(self.n_items):
            users_rating, ratings = self.get_users_who_rate_item(item)
            W_item = self.W[:, users_rating]

            grad = -(ratings - self.X[item, :].dot(W_item)).dot(W_item.T) / \
                self.n_ratings + self.regularization*self.X[item, :]
            self.X[item, :] -= self.eta*grad.reshape(self.latent)

    def updateW(self):
        for user in range(self.n_users):
            items_rated, ratings = self.get_items_rated_by_user(user)
            X_user = self.X[items_rated, :]

            grad = -X_user.T.dot(ratings - X_user.dot(
                self.W[:, user]))/self.n_ratings + self.regularization*self.W[:, user]
            self.W[:, user] -= self.eta*grad.reshape(self.latent)

    def fit(self):
        self.normalize()
        for _ in range(self.max_iterations):
            self.updateX()
            self.updateW()
            # if (it + 1) % self.print_every == 0:
            #     rmse_train = self.evaluate_RMSE(self.Y_raw_data)
            #     print('iter =', it + 1, ', loss =',
            #           self.loss(), ', RMSE train =', rmse_train)

    def __pred(self, u, i):
        u = int(u)
        i = int(i)
        # if self.based == USER:
        #     bias = self.hat[u]
        # else:
        #     bias = self.hat[i]
        # pred = self.X[i, :].dot(self.W[:, u]) + bias
        pred = self.X[i, :].dot(self.W[:, u])

        
        # return self.X[i, :].dot(self.W[:, u])

        if pred < 0:
            return 0
        if pred > 5:
            return 5
        return pred

    def recommend(self, u):
            index_user_rated = np.where(self.Y_raw_data[:, 0] == u)[0]
            items_rated_by_user = self.Y_raw_data[index_user_rated, 1].astype(np.int32)

            index_user_rated_test = np.where(self.Y_test[:, 0] == u)[0]
            items_rated_by_user_test = self.Y_test[index_user_rated_test, 1].astype(np.int32)

            items_rated_by_user_all = items_rated_by_user.tolist() + items_rated_by_user_test.tolist()

            recommended_items = []
            pred = self.X.dot(self.W[:, u])
            for i in range(self.n_items):
                if i not in items_rated_by_user_all:
                    # if pred[i] > 0:
                    #     recommended_items.append(i)
                    rating = pred[i]
                    recommended_items.append([i,rating])
            temp = np.array(recommended_items)
            index = np.argsort(temp[:, 1])[-self.amount:]
            return temp[index,0].astype(np.int32)[::-1]
            # return recommended_items

    def recommendation_result(self):
        result = []
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            row = [u] + recommended_items.tolist()
            result.append(row)
        return result  

    def evaluate(self):
        actual = self.Y_test[:, 2]
        predicted = []
        n_tests = self.Y_test.shape[0]
        for n in range(n_tests):
            predicted.append(self.__pred(self.Y_test[n, 0], self.Y_test[n, 1]))

        MSE = mean_squared_error(actual, predicted)
        RMSE = math.sqrt(MSE)

        MAE = mean_absolute_error(actual, predicted)
        return RMSE, MAE

# def get_data(data):
#     ratings = pd.read_csv(data, sep = "[;, \t]", encoding='latin-1',on_bad_lines='skip',engine='python',usecols=[0,1,2], header=None)
#     Y_data = ratings.to_numpy()
#     return Y_data


# parent_path = pathlib.Path(__file__).parent.resolve()
# path = os.path.join(parent_path,'data-150-binary.csv')
# x = get_data(path)
# Ydata, Ytest = train_test_split(x)

# rs_cf_cosine_user = MatrixFactorizattion(Y_data = Ydata, Y_test= Ytest)
# rs_cf_cosine_user.fit()
# print(rs_cf_cosine_user.recommendation_result())
# print(rs_cf_cosine_user.evaluate())
# print(rs_cf_cosine_user.n_users, rs_cf_cosine_user.n_items)

# parent_path = pathlib.Path(__file__).parent.parent.resolve()
# path = os.path.join(parent_path,'uploads', '2022-11-20_091046.383574-data.csv')
# x = get_data(path)
# Y_train, Y_test = train_test_split(x)
# print(x)
# r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

# ratings_base = pd.read_csv(os.path.join(
#     parent_path, 'ub.base'), sep='\t', names=r_cols, encoding='latin-1')
# ratings_test = pd.read_csv(os.path.join(
#     parent_path, 'ub.base'), sep='\t', names=r_cols, encoding='latin-1')

# rate_train = ratings_base.to_numpy()
# rate_test = ratings_test.to_numpy()

# # indices start from 0
# rate_train[:, :2] -= 1
# rate_test[:, :2] -= 1

# rs = MatrixFactorizattion(Y_train, Y_test=Y_test, latent=10, regularization=.1, eta=0.70, max_iterations=100)
# rs.fit()
# rs.normalize()
# print(Y_train)
# print(Y_test)


# print(rs.evaluate(Y_test))
# print(rs.evaluate(Y_test))
# print(rs.recommendation_result())
# print(rs.evaluate())



# r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

# parent_path = pathlib.Path(__file__).parent.resolve()
# path_base = os.path.join(parent_path, 'ub.base')
# path_test = os.path.join(parent_path, 'ub.test')

# ratings_base = pd.read_csv(path_base, sep='\t', names=r_cols, encoding='latin-1')
# ratings_test = pd.read_csv(path_test, sep='\t', names=r_cols, encoding='latin-1')

# rate_train = ratings_base.to_numpy()
# rate_test = ratings_test.to_numpy()

# # indices start from 0
# rate_train[:, :2] -= 1
# rate_test[:, :2] -= 1
# # print(rate_train)
# # rs = MatrixFactorizattion(x, latent=10, regularization=.1, print_every=10,
# #                           eta=0.70, max_iterations=100, based=ITEM)
# # rs = MatrixFactorizattion(rate_train, latent = 30, regularization=.1, eta=0.75, based=USER, max_iterations=100,)
# # rs.fit()
# # print(rs.evaluate(rate_test))
# print(rate_train)

