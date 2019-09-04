from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib
import pyodbc
import pandas as pd


class DatabaseConnection:
    def __init__(self):
        self.cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                              "Server=S-KV-CENTER-S27;"
                              "Database=4t.Dev;"
                              "Trusted_Connection=yes;")

    def get_filials_for_articule(self, articule, start_date, end_date, min_days=0):
        sql_query_select_filials_for_articule = f"select FilialId \
                from [SalesHub.Dev].[DataHub].[v_SalesStores] \
                where LagerId={articule} and [Date] >= {start_date} and [Date] <= {end_date} and QtySales > 0\
                group by FilialId  \
                having count(distinct [Date]) >= {min_days}"
        df = pd.read_sql_query(sql_query_select_filials_for_articule, self.cnxn)
        return set(sorted(df['FilialId'].tolist()))

    def get_all_sales_by_articule(self, articule, start_date, end_date):
        sql_query_all_sales_by_articule = f"select FilialId,\
                [Date],\
                sum(QtySales) as quantity, \
                case when sum(QtySales) = 0 then 0 else sum(AmountSales) / sum(QtySales) end as mean_price  \
                from [SalesHub.Dev].[DataHub].[v_SalesStores] \
                where LagerId = {articule} and [Date] >= {start_date} and [Date] <= {end_date}\
                group by FilialId, [Date]"
        df = pd.read_sql_query(sql_query_all_sales_by_articule, self.cnxn)
        df.fillna({'quantity': 0}, inplace=True)
        df.fillna({'mean_price': 0}, inplace=True)
        return df

    def get_all_sales_by_articule_and_filial(self, articule, filial, start_date, end_date):
        sql_query_sales_by_articule_and_filial = f"select [Date], \
                sum(QtySales) as quantity, \
                case when sum(QtySales) = 0 then 0 else sum([AmountSales]) / sum(QtySales) end as mean_price  \
                from [SalesHub.Dev].[DataHub].[v_SalesStores] \
                where LagerId = {articule}\
                and FilialId = {filial}\
                and [Date] >= {start_date} and [Date] <= {end_date}\
                group by [Date]"
        df = pd.read_sql_query(sql_query_sales_by_articule_and_filial, self.cnxn)
        df = df.sort_values(by='Date')
        return df

    def get_special_days(self, articule, filial, start_date, end_date):
        start = datetime.datetime.strptime(start_date, "'%Y-%m-%d'").date()
        end = datetime.datetime.strptime(end_date, "'%Y-%m-%d'").date()

        sql_query_promo_days = f"select date_id\
            from [4t.Dev].[4t_data].[promos]\
            where product_id={articule}\
            and store_id = {filial}\
            and date_id >={start_date} and date_id <={end_date}"
        promo_days = pd.read_sql_query(sql_query_promo_days, self.cnxn)['date_id'].tolist()

        start_year = start.year
        end_year = end.year

        # adding New Year holidays to ignore list
        for year in range(start_year, end_year + 1):
            promo_days.append(datetime.date(year, 1, 1))
            promo_days.append(datetime.date(year, 1, 2))
            promo_days.append(datetime.date(year, 1, 3))
            promo_days.append(datetime.date(year, 12, 28))
            promo_days.append(datetime.date(year, 12, 29))
            promo_days.append(datetime.date(year, 12, 30))
            promo_days.append(datetime.date(year, 12, 31))
            promo_days.append(datetime.date(year, 3, 7))
            promo_days.append(datetime.date(year, 3, 8))
            if year == 2017:
                promo_days.append(datetime.date(2017, 4, 14))
                promo_days.append(datetime.date(2017, 4, 15))
                promo_days.append(datetime.date(2017, 4, 16))
            if year == 2018:
                promo_days.append(datetime.date(2018, 4, 6))
                promo_days.append(datetime.date(2018, 4, 7))
                promo_days.append(datetime.date(2018, 4, 8))
            if year == 2019:
                promo_days.append(datetime.date(2019, 4, 26))
                promo_days.append(datetime.date(2019, 4, 27))
                promo_days.append(datetime.date(2019, 4, 28))

        return promo_days


def get_df_without_empty_dates(df, start_date, end_date):

    start = datetime.datetime.strptime(start_date, "'%Y-%m-%d'").date()
    end = datetime.datetime.strptime(end_date, "'%Y-%m-%d'").date()

    m = (end - start).days + 1

    old_dates = df['Date'].values
    old_quantities = df['quantity'].values
    old_prices = df['mean_price'].values

    new_dates = np.array([start + datetime.timedelta(days=i) for i in range(m)])
    new_quantities = np.zeros(m)
    new_prices = np.zeros(m)

    i = 0
    for j in range(len(old_dates)):
        if old_dates[j] > end:
            break
        while new_dates[i] < old_dates[j]:
            i += 1
        if new_dates[i] == old_dates[j]:
            new_quantities[i] = old_quantities[j]
            new_prices[i] = old_prices[j]

    current_price = 0

    for i in range(m):
        if new_quantities[i] > 0:
            current_price = new_prices[i]
        else:
            new_prices[i] = current_price

    df_new = pd.DataFrame(columns=['Date', 'quantity', 'mean_price'], index=np.arange(m))

    df_new['Date'] = new_dates
    df_new['quantity'] = new_quantities
    df_new['mean_price'] = new_prices

    return df_new


def add_special_days_column_to_df(df, special_days):
    dates = df['Date'].values
    is_special_day = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if dates[i] in special_days:
            is_special_day[i] = True
    df['is_special_day'] = is_special_day


def get_tuned_df(df, special_days, start_date, end_date):
    df_without_empty_days = get_df_without_empty_dates(df, start_date, end_date)
    add_special_days_column_to_df(df_without_empty_days, special_days)
    return df_without_empty_days


def get_mean_error(df, predictions):
    non_nan_indices = ~np.isnan(predictions)
    num_predicted_items = np.sum(non_nan_indices)

    if num_predicted_items > 0:
        total_squared_error = np.sum((df['quantity'].values[non_nan_indices] - predictions[non_nan_indices]) ** 2)
        mean_error = (total_squared_error / num_predicted_items) ** 0.5
    else:
        mean_error = np.nan

    return mean_error


def get_data_arrays(articule, start_date, end_date, filials_set=None):

    if filials_set is None:
        filials_set = database_connection.get_filials_for_articule(articule, start_date, end_date, min_days=30)

    df_sales = database_connection.get_all_sales_by_articule(articule, start_date, end_date)

    list_x = []
    list_y = []

    for filial in filials_set:
        df = df_sales.loc[df_sales['FilialId'] == filial]
        df = df.sort_values(by='Date')
        # get list of special days, such as when there were promos or New Year holidays
        special_days = database_connection.get_special_days(articule, filial, start_date, end_date)
        df = get_tuned_df(df, special_days, start_date, end_date)

        sales_array = df['quantity'].values
        special_days_array = df['is_special_day'].values

        num_rows = len(sales_array)

        for i in range(num_prev_days, num_rows):
            if np.sum(special_days_array[i - num_prev_days: i + 1]) == 0:
                list_x.append(sales_array[i - num_prev_days: i])
                list_y.append(df.at[i, 'quantity'])

    array_x = np.array(list_x)
    array_y = np.array(list_y)

    array_x = np.reshape(array_x, (array_x.shape[0], array_x.shape[1], 1))

    return array_x, array_y


make_plots = True

start_train_date = "'2017-01-07'"
end_train_date = "'2018-01-07'"

start_test_date = "'2018-01-08'"
end_test_date = "'2018-08-26'"

num_prev_days = 14

articule = 32485

database_connection = DatabaseConnection()

print("Collecting data for training ANN...")
array_x_train, array_y_train = get_data_arrays(articule, start_train_date, end_train_date)
print("Colected data")

x_max = np.max(array_x_train)
print("x_max=", x_max)

array_x_train /= x_max
array_y_train /= x_max

model = Sequential()
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

print("Fitting the model...")

start_time = time.time()

model.fit(array_x_train, array_y_train, epochs = 20, batch_size = 32)

print("Finished model fitting!")
seconds = time.time() - start_time

minutes = seconds / 60
print(f"Model fitting took {seconds:.2f} seconds")

test_filials_set = {445, 1934, 1935, 2014, 2034, 2015, 2049, 2225, 2279, 2315, 2473, 2616}

for filial in test_filials_set:
    df = database_connection.get_all_sales_by_articule_and_filial(articule, filial, start_test_date, end_test_date)
    df = df.sort_values(by='Date')
    special_days = database_connection.get_special_days(articule, filial, start_test_date, end_test_date)
    df = get_tuned_df(df, special_days, start_test_date, end_test_date)
    sales_array = df['quantity'].values

    y_predicted = np.array([np.nan] * len(df))
    for i in range(60, len(df)):
        X_for_test = sales_array[i - num_prev_days: i]
        X_for_test = np.reshape(X_for_test, (1, X_for_test.shape[0], 1))
        X_for_test = X_for_test / x_max
        y_predicted[i] = model.predict(X_for_test)[0][0] * x_max

        if df.at[i, 'is_special_day']:
            y_predicted[i] = np.nan

    print(f"Filial {filial}")

    mean_error = get_mean_error(df, y_predicted)
    print(f'Mean error for RNN on filial {filial} is {mean_error}')

    if make_plots:
        font = {'weight': 'bold', 'size': 25}
        matplotlib.rc('font', **font)

        plt.title(f'Продажи артикула {articule} на филиале {filial}')
        axes = plt.gca()
        plt.plot(df['Date'], df['quantity'], marker='o', markersize=6, linewidth=4, label='Sales')
        plt.plot(df['Date'], y_predicted, marker='o', markersize=6, linewidth=4, label='RNN prediction')
        plt.legend()
        plt.grid()
        plt.show()

