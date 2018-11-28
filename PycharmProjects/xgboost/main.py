import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from  sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import pandas as pd

def make_pred(y_train_predict):
    row_idx = -1
    predict = np.zeros((np.size(y_train_predict, 0), 1))
    for row in y_train_predict:
        row_idx += 1
        for i in range(0, np.size(row, 0)):
            if (row[i] == 1):
                predict[row_idx] = i
    return predict


def make_pred_test(y_test_predict, df_test):
    result_pred = np.zeros((0, 1))
    j = 0
    pred_i = -1
    test_idx = np.zeros((0, 1))
    for i in df_test['SK_ID']:
        cur_i = i
        if cur_i != pred_i:
            mask = df_test['SK_ID'] == i # значения которые надо усреднять
            temp_sum = sum(y_test_predict[mask])
            temp_mean = temp_sum / sum(mask)
            #print(temp_mean)
            result_pred = np.append(result_pred, temp_mean)
            # make test idx sk_id
            test_idx = np.append(test_idx, i)
            j += 1
        pred_i = cur_i
    print(j)
    return result_pred, test_idx


def find(y_test1, i, test_idx):
    k = 0
    for c in test_idx:
        if c == i:
            return y_test1[k]
        k += 1



df = pd.read_csv('C:/Users/Artem/Desktop/telecom/dataset/bs_avg_kpi.csv', sep=';', decimal=',', index_col=False, dtype='float64',
                 usecols=range(1, 45, 1), chunksize=1200000, iterator=True)

df1 = pd.read_csv('C:/Users/Artem/Desktop/telecom/dataset/train/subs_bs_consumption_train.csv', sep=';', decimal=',', index_col=False, dtype='float64',
                 usecols=[0, 1, 3, 4, 5, 6], chunksize=120000, iterator=True)

df3 = pd.read_csv('C:/Users/Artem/Desktop/telecom/dataset/train/subs_bs_data_session_train.csv', sep=';', decimal=',', index_col=False, dtype='float64',
                 usecols=[0, 1, 2, 4], iterator=True)

df4 = pd.read_csv('C:/Users/Artem/Desktop/telecom/dataset/train/subs_bs_voice_session_train.csv', sep=';', decimal=',', index_col=False, dtype='float64',
                 usecols=[0, 1, 2, 4], iterator=True)

ans = pd.read_csv('C:/Users/Artem/Desktop/telecom/dataset/train/subs_csi_train.csv', sep=';', decimal=',', index_col=False, dtype='float64',
                 usecols=[0, 1])

ans.set_index('SK_ID', inplace=True,
              drop=False) # создаю новые индексы по айди
#print(ans)
train = pd.read_csv('C:/Users/Artem/Desktop/telecom/dataset/train/subs_features_train.csv', sep=';', decimal=',', index_col=False, dtype='float64',
                 usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])

test = pd.read_csv('C:/Users/Artem/Desktop/telecom/dataset/test/subs_features_test.csv', sep=';', decimal=',', index_col=False, dtype='float64',
                 usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])

test_idx = pd.read_csv('C:/Users/Artem/Desktop/telecom/dataset/test/subs_csi_test.csv', sep=';', decimal=',', index_col=False, dtype='float64',
                 usecols=[0])

#print(np.shape(test))

# Заполняю на средними знач
train = train.fillna(train.median(axis=0), axis=0)
test = test.fillna(test.median(axis=0), axis=0)
test = test.sort_values(by=['SK_ID'])

#profile_train = train.describe()
#profile_ans = ans.describe()


#print(test)
#print(profile_train.to_string())
j = 0
# делаю игрик треин
y_train = np.zeros((train.shape[0], 1))
print(train.shape[0])
for i in train['SK_ID']:
    y_train[j] = ans['CSI'][i]
    j += 1

#print(np.sum(y_train))

# начало алгоритма
# препроцессинг параметров

from sklearn.neural_network import MLPClassifier
tmp = np.zeros((np.size(y_train, 0), 2))
idx_i = 0
idx = y_train == 0
for i in idx:
    if (i):
        tmp[idx_i, 0] = 1
    idx_i += 1

idx_i = 0
idx = y_train == 1
for i in idx:
    if (i):
        tmp[idx_i, 1] = 1
    idx_i += 1


y_train = tmp
#print(y_train)
X_train = np.array(train)
X_ans = np.array(test)
Scale = preprocessing.StandardScaler().fit(X_train)
X_train = Scale.transform(X_train)

Scale = preprocessing.StandardScaler().fit(X_ans)
X_ans = Scale.transform(X_ans)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3, random_state = 11)

print(np.shape(y_test))

####################################################################


clf = MLPClassifier(activation='tanh', alpha=1e-5, batch_size=1,
                    beta_1=0.9, beta_2=0.999, early_stopping=False,
                    epsilon=1e-08, hidden_layer_sizes=(100, 5),
                    learning_rate='constant', learning_rate_init=0.00001,
                    max_iter=1200, momentum=0.9,
                    nesterovs_momentum=True, power_t=0.5, random_state=15,
                    shuffle=True, solver='adam', tol=0.0001,
                    validation_fraction=0.1, verbose=True, warm_start=False)


clf.fit(X_train, y_train)


################-make_my test prediction-####################

y_train_predict = clf.predict(X_train)
y_train_predict = make_pred(y_train_predict)

y_test_predict = clf.predict(X_test)
y_test_predict = make_pred(y_test_predict)

y_train = make_pred(y_train)
y_test = make_pred(y_test)

print(np.shape(y_train_predict))
print(np.shape(y_train))

err_train = np.mean(y_train_predict == y_train)
err_test = np.mean(y_test_predict == y_test)

print('test_p', err_test)
print('train_p', err_train)

err_train = f1_score(y_train, y_train_predict, average='macro')
err_test = f1_score(y_test, y_test_predict, average='macro')

print('test_f1', err_test)
print('train_f1', err_train)

err_train = roc_auc_score(y_train, y_train_predict, average='macro')
err_test = roc_auc_score(y_test, y_test_predict, average='macro')

print('test_auc', err_test)
print('train_auc', err_train)

###################-make_their test prediction-###################


y_test_predict = clf.predict(X_ans)
y_test_predict = make_pred(y_test_predict)

y_test_predict, test_idx_pred = make_pred_test(y_test_predict, test)
#print(test_idx_pred)

# save without percent
df = pd.DataFrame(y_test_predict)
pd.options.display.float_format = '{:,.0f}'.format
df = df.astype(float)
df.to_csv('save_res.csv', index=False, header=True)
np.savetxt(r'C:/Users/Artem/Desktop/save_result.txt', df.values, fmt='%d')

# percent

#y_test_predict = y_test_predict > 0.5 # если больше то 1
test_idx_arr = np.array(test_idx)

# correcting idx in answer
new_test_y_predict = np.zeros((0, 1))

for v in test_idx_arr:
    new_test_y_predict = np.append(new_test_y_predict, find(y_test_predict, v, test_idx_pred))

df = pd.DataFrame(new_test_y_predict)
pd.options.display.float_format = '{:,.0f}'.format
df = df.astype(float)
df.to_csv('submit.csv', index=False, header=True)
np.savetxt(r'C:/Users/Artem/Desktop/result.txt', df.values, fmt='%d')


