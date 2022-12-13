import math
import os
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

class Popular():
    def __init__(self, Y_data, Y_test, k = 10):
        self.Y_data = Y_data
        self.Y_test = Y_test
        self.n_users_test = int(np.max(self.Y_test[:, 0])) + 1 
        self.n_items_test = int(np.max(self.Y_test[:, 1])) + 1 
        self.k = k
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 if self.n_users_test<int(np.max(self.Y_data[:, 0])) + 1 else self.n_users_test
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1 if self.n_items_test<int(np.max(self.Y_data[:, 1])) + 1 else self.n_items_test

    def normalized(self):
        items = self.Y_data[:,1]
        self.hat = np.zeros(self.n_items,)
        for item in range(self.n_items):
            index_item_rated = np.where(items == item)[0]
            ratings = self.Y_data[index_item_rated, 2]
            if len(ratings) == 0: average_item_rated = 0
            else: average_item_rated = np.mean(ratings) 
            if np.isnan(average_item_rated):
                average_item_rated = 0
            self.hat[item] = average_item_rated

    def pred(self,u,i):
        i = int(i)
        return self.hat[i]

    def recommend(self, u):
        index_user_rated = np.where(self.Y_data[:, 0] == u)[0]
        index_user_rated_test = np.where(self.Y_test[:, 0] == u)[0]

        item_rated_by_users = self.Y_data[index_user_rated, 1].astype(np.int32)
        item_rated_by_users_test = self.Y_test[index_user_rated_test, 1].astype(np.int32)

        item_rated_by_users_all = item_rated_by_users.tolist() + item_rated_by_users_test.tolist()

        items_sorted = np.argsort(self.hat).tolist()
        for u in item_rated_by_users_all[:]:
            items_sorted.remove(u)
        return items_sorted[-self.k:]

    def fit(self):
        self.normalized()

    def recommendation_result(self):
        result = []
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            recommended_items.reverse()
            row = [u] + recommended_items
            result.append(row)
        return result

    def evaluate(self):
            actual = self.Y_test[:, 2]
            predicted = []
            n_tests = self.Y_test.shape[0]
            for n in range(n_tests):
                predicted.append(self.pred(self.Y_test[n, 0], self.Y_test[n, 1]))

            MSE = mean_squared_error(actual, predicted)
            RMSE = math.sqrt(MSE)

            MAE = mean_absolute_error(actual, predicted)
            return RMSE, MAE

# def get_data(path):
#     ratings = pd.read_csv(path, sep = "[;, \t]", header=None, encoding='latin-1', engine='python')
#     Y_data = ratings.to_numpy()
#     return Y_data

# parent_path = pathlib.Path(__file__).parent.resolve()
# path = os.path.join(parent_path,'data-r.csv')
# x = get_data(path)
# print(path)


# parent_path = pathlib.Path(__file__).parent.parent.resolve()
# path = os.path.join(parent_path,'uploads','2022-11-20_091046.383574-data.csv')
# x = get_data(path)
# Ydata, Ytest = train_test_split(x)
# print(x)

# rs = Popular(Y_data= Ydata, Y_test= Ytest)
# rs.fit()
# print(rs.evaluate(Ytest))

