%load_ext autoreload
%autoreload 2

import torch
import numpy as np
import pandas as pd
import plotly_express as px

import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm
# 프로그램의 진행상황을 확인하게 해주는 모듈
tqdm.pandas()

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (12.0, 12.0)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ACF(Auto Correlation Function)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import plotly.graph_objects as go

import pickle # 텍스트 상태의 데이터가 아닌 파이썬 객체 자체를 파일로 저장하는 모듈

import traceback
import tqdm
import numpy as np
import pandas as pd
from functools import partial
from tqdm.contrib.concurrent import process_map
from collections import defaultdict

tqdm.tqdm().pandas()

from torch.utils.data import Dataset, DataLoader

# 5년간의 store-item의 판매 데이터
# 10개의 상점에서 50개의 다른 품목에 대한 3개월의 판매를 예측하는 것이 목표

# 데이터 전처리
train = pd.read_csv(r'C:\Users\kijeong\research\sales_data/train.csv')
test = pd.read_csv(r'C:\Users\kijeong\research\sales_data/test.csv')

train.describe()
print(train.shape) # 913000, 4
train.head()

test.describe()
print(test.shape) # 45000, 4
test.head()

train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

print(train['date'].min(), train['date'].max()) # 2013~2017년
print(test['date'].min(), test['date'].max()) # 2018년 1~3월

test['sales'] = np.nan
data = pd.concat([train, test], ignore_index=True)
data['store_item_id'] = data['store'].astype(str) + '_' + data['item'].astype(str)
# astype(str)은 type을 string으로 변경하라는 의미
data['store_item_id'].describe()
# count : 958000
# unique : 50 x 10 = 500
# 모두 1916개로 같음

data['dayofweek'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['day'] = data['date'].dt.day

data['year'].min(), data['year'].max() # 2013, 2018
data['year_mod'] = (data['year'] - data['year'].min()) / (data['year'].max() - data['year'].min())
data['year_mod'] 
# (2013, 2014, 2015, 2016, 2017, 2018) = (0, 0.2, 0.4, 0.6, 0.8, 1)

# 주기에 대한 정보를 넣기 위해서 sin, cos으로 변환한 값 추가
# 1주는 7일 주기, 한 달은 30일 주기, 1년은 12달 주기
# year는 주기가 없으므로 추가하지 않음
def sin_transform(values):
    return np.sin(2*np.pi*values/len(set(values)))

def cos_transform(values):
    return np.cos(2*np.pi*values/len(set(values)))

data['dayofweek_sin'] = sin_transform(data['dayofweek'])
data['dayofweek_cos'] = cos_transform(data['dayofweek'])
data['month_sin'] = sin_transform(data['month'])
data['month_cos'] = cos_transform(data['month'])
data['day_sin'] = sin_transform(data['day'])
data['day_cos'] = cos_transform(data['day'])

data.drop('id', axis=1, inplace=True)
data = data.sort_values(['store_item_id', 'date'])
data

plt.plot(sin_transform(np.arange(0,12)), label='month_sin')
plt.plot(cos_transform(np.arange(0,12)), label='month_cos')
plt.legend()


def plot_line(acf):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(acf))), y=acf,
                        mode='lines',
                        name='acf'))
    fig.show()

# 여기서부터는 training data 사용
train['store_item_id'] = train['store'].astype(str) + '_' + train['item'].astype(str)

item_data = train[train['store_item_id'] == '1_1'].sort_values('date')
# 1번 상점, 1번 상품에 대한 데이터를 날짜순으로 정렬
 
avg_data = train[['date', 'sales']].groupby('date').mean()
# train data에서 날짜와 판매량만 가져와서 날짜를 기준으로 그룹화한 후 평균 계산
# 즉 해당 날짜에 판매된 모든 상점, 모든 제품의 판매량의 평균을 계산

item_ac = acf(item_data['sales'], nlags=366)
# 0번째 시차(로_0 = 1)도 보여줌 -> 총 길이가 367
avg_ac = acf(avg_data['sales'], nlags=len(avg_data))
# 전체 판매량의 평균에 대한 acf를 계산
# len(avg_data) = 1826이므로 모든 시차에 대해서 acf 전부 계산
# 이 때 len(avg_ac)도 1826이 나오는데 이는 1826이전 시차와는 계산이 불가능하기 때문에 하나가 빠져서 이렇게 된거임
len(item_ac) # 367
len(avg_ac) # 1826

px.line(item_data, x='date', y='sales') # 계절성이 뚜렷함을 확인할 수 있음
plot_line(item_ac)
plot_line(avg_ac)

# 오름차순 정렬한 뒤 그 것의 인덱스를 100개까지 반환
important_lags = np.argsort(-avg_ac)[:100]
important_lags

# 서로 다른 품목 10개를 랜덤하게 뽑아서 sample data를 만듦
sample_data = data[data['store_item_id'].isin(pd.Series(data['store_item_id'].unique()).sample(10))]
px.line(sample_data, x='date', y='sales', color='store_item_id', title='Store item sales')


# Normalize data

train['store_item_id'] = train['store'].astype(str) + '_' + train['item'].astype(str)

#mode = 'valid'
mode = 'test'

if mode == 'valid':
    scale_data = train[train['date'] < '2017-01-01']
else:
    scale_data = train[train['date'] >= '2014-01-01']

scale_data # 2014~2017년 데이터, test데이터

# Get yearly autocorrelation for each timeseries
def get_yearly_autocorr(data):
    ac = acf(data, nlags=366)
    return (0.5 * ac[365]) + (0.25 * ac[364]) + (0.25 * ac[366])


scale_map = {}
scaled_data = pd.DataFrame()


def save_scale_map(name, scale_map):
    with open(name + '.pickle', 'wb') as handle:
        pickle.dump(scale_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

data
scale_data
# 아래 for문에 대한 설명
# 원 데이터를 가져와서 각 매장의 각 상품별로 데이터를 뽑음
# store_item_id에는 값 하나가 들어가고, 그에 대한 데이터가 item_data에 저장
# scale_data에서 위에서 뽑은 store_item_id와 일치할 경우만 판매량을 뽑아서 판매량의 평균, 표준편차, 연도별 ACF를 구함
# 원 데이터에서 뽑은 item_data에서 판매량을 뽑아서 정규화
# scale_map에 원 데이터 판매량의 평균과 표준편차를 dictinary형태로 저장
# item_data에 평균과 연도별 ACF를 새로운 column으로 추가
# 모든 item_data를 scaled_data에 하나씩 쌓아서 최종 데이터프레임 완성
for store_item_id, item_data in tqdm(data.groupby('store_item_id', as_index=False)):
    sidata = scale_data.loc[scale_data['store_item_id'] == store_item_id, 'sales']
    mu = sidata.mean()
    sigma = sidata.std()
    yearly_autocorr = get_yearly_autocorr(sidata)
    item_data.loc[:,'sales'] = (item_data['sales'] - mu) / sigma
    scale_map[store_item_id] = {'mu': mu, 'sigma': sigma}
    item_data['mean_sales'] = mu
    item_data['yearly_corr'] = yearly_autocorr
    scaled_data = pd.concat([scaled_data, item_data], ignore_index=True)

scaled_data['yearly_corr'] = ((scaled_data['yearly_corr'] - scaled_data['yearly_corr'].mean()) / scaled_data['yearly_corr'].std())
scaled_data['mean_sales'] = (scaled_data['mean_sales'] - scaled_data['mean_sales'].mean()) / scaled_data['mean_sales'].std()
scaled_data.loc[scaled_data['store_item_id'] == '1_1', 'sales'].plot()
# 각 품목별 연도별ACF와 판매량 평균도 다시 한 번 정규화 시켜줌
scaled_data

# pass...
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

scaled_data.drop(['date','day', 'month', 'year', 'dayofweek', 'mean_sales', 'store_item_id'], axis=1).head()
scaled_data = reduce_mem_usage(scaled_data)
scaled_data
scaled_data.to_pickle(r'C:\Users\kijeong\research\sales_data/processed_data_test_stdscaler.pkl')
# due to a weird windows problem, multiprocessing cannot be run in notebook
# Run sequence builder on scaled_data, and load the pickle here

scaled_data = pd.read_pickle(r'C:\Users\kijeong\research\sales_data/processed_data_test_stdscaler.pkl')

### Sequence_builder.py run ###

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def split_sequence_difference(group_data, n_steps_in, n_steps_out, x_cols, y_col, diff, additional_columns):
    try:
        X, y = list(), list()
        additional_col_map = defaultdict(list) # 디폴트값이 리스트인 딕셔너리, 값을 지정해주지 않으면
        group_data[y_col] = group_data[y_col].diff()
        additional_col_map['x_base'] = []
        additional_col_map['y_base'] = []
        additional_col_map['mean_traffic'] = []
        for i in range(diff, len(group_data)):
            # find the end of this pattern
            x_base = group_data.iloc[i - 1]['unmod_y']
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(group_data)-1:
                break
            y_base = group_data.iloc[end_ix - 1]['unmod_y']
            # gather input and output parts of the pattern
            if len(x_cols) == 1:
                x_cols = x_cols[0]
            seq_x, seq_y = group_data.iloc[i:end_ix, :][x_cols].values, group_data.iloc[end_ix:out_end_ix, :][y_col].values
            for col in additional_columns:
                additional_col_map[col].append(group_data.iloc[end_ix][col])
            additional_col_map['x_base'].append(x_base)
            additional_col_map['y_base'].append(y_base)
            additional_col_map['mean_traffic'] = group_data['unmod_y'].mean()
            X.append(seq_x)
            y.append(seq_y)
        additional_column_items = sorted(additional_col_map.items(), key=lambda x: x[0])
        return (np.array(X), np.array(y), *[i[1] for i in additional_column_items])
    except Exception as e:
        print(e)
        print(group_data.shape)
        traceback.print_exc()


# 아래 lag_fns for문에 대해서 실행해보기
lag_fns = [last_year_lag]
split_dfs[0].sort_values('date')
group_data = split_dfs[0].groupby('store_item_id')
group_data
group_data2 = split_dfs[0].sort_values('date').groupby('store_item_id')

'''for i, lag_fn in enumerate(lag_fns):
    group_data[f'lag_{i}'] = lag_fn(group_data2[y_cols[0]])
group_data2[y_cols[0]] # 안돌아감 

split_dfs[1] # 아직 변환하지 않은 값, 변환 후와 비교하기 위해서 뒀음
split_dfs[0].lag_0.head(367) # 366개의 데이터가 밀려서 NaN이 됐음 '''
# 결론 : group전체로 밖에서 돌리는거는 불가능 -> group 하나만 돌려보기
# store_item_id가 1_1인 애들만 돌려보기
t_df = split_dfs[0]
y_cols[0] # sales
group_data_1_1 = t_df[ t_df.store_item_id == '1_1'].sort_values('date')
# 위 group_data_1_1는 그룹 하나에 대해서 sort한 결과
for i, lag_fn in enumerate(lag_fns):
    group_data_1_1[f'lag_{i}'] = lag_fn(group_data_1_1[y_cols[0]])
group_data_1_1 # sales를 1년뒤로 밀고, 그 앞뒤 데이터를 이용해서 가중평균한 값을 lag_0에 저장
step=1
steps = list(range(0, len(group_data_1_1), step)) # 0~1915
len(steps) # 1916
n_steps_in=180
n_steps_out=90
additional_columns = ['store_item_id', 'item', 'store', 'date', 'yearly_corr']
additional_col_map = defaultdict(list)
X, y = list(), list()
len(group_data_1_1)

for i in steps:
    # find the end of this pattern
    end_ix = i + n_steps_in # 180, 181, 182, ... , 2095
    out_end_ix = end_ix + n_steps_out # 270, 271, 272, ... , 1916
    # check if we are beyond the dataset
    if out_end_ix > len(group_data_1_1): # 실제 분할하기 전이기 때문에 len(group_data)는 83이 아니라 83*50임
        break
    # gather input and output parts of the pattern
    if len(x_cols) == 1:
        x_cols = x_cols[0]
    seq_x, seq_y = group_data_1_1.iloc[i:end_ix, :][x_cols].values, group_data_1_1.iloc[end_ix:out_end_ix, :][y_cols + [f'lag_{i}' for i in range(len(lag_fns))]].values
    # seq_x에는 처음 180개의 행과 x_cols에 해당하는 8개의 열로 이루어진 값들을 넣어줌
    # seq_y에는 seq_x에 들어간 180개의 행 다음 행부터 90개의 행을 넣어줌
    # 이 때 열은 y_cols에 해당하는 8개의 열과 위에서 새로 만든 lag_0열, 즉 9개의 열이 들어감
    # 여기서는 90개를 밀어서 그렇지만, 나중에는 lag_0에 NaN이 아니라 값이 들어가있을거임
    for col in additional_columns:
        additional_col_map[col].append(group_data_1_1.iloc[end_ix][col])
    # additional_columns, 즉 ['store_item_id', 'item', 'store', 'date', 'yearly_corr']에 해당하는 열을
    # 하나씩 col에 넣고, 각 col에 해당하는 값들을 전부 리스트 안에 넣어줌
    # 결과적으로 additional_col_map에는 additional_columns에 해당하는 모든 값들이 들어갈거임 
    X.append(seq_x)
    y.append(seq_y)
    # seq_x와 seq_y로 구한 값들을 차례대로 X와 y에 넣어줌
additional_column_items = sorted(additional_col_map.items(), key=lambda x: x[0])
# key=lambda x: x[0]로 지정하면 정렬할 때 첫번째 값을 기준으로 오름차순으로 정렬함
# items()는 딕셔너리에서 key와 value의 쌍을 반환해줌
len(additional_column_items) # 총 열이 5개였으므로 길이도 5
additional_column_items[0] # date
additional_column_items[0][1] # date의 value, 길이는 1647
# 1647이라는 숫자는 1646번째 인덱스, 즉 1647번째 열에서부터 180개를 가지고 90개를 예측하면 모든게 다 뽑히고, 그 다음 break가 되게 설정해뒀기 때문 
additional_column_items[1] # item
additional_column_items[4] # yearly_corr
np.array(X).shape # 1647, 180, 8
# 각 행에서부터 180이전 시차까지의 값들이 (180, 8)의 array형태로 총 1647개가 있는 것
np.array(y).shape # 1647, 90, 9
# 각 행에서부터 180이전 시차까지의 값들이 (90, 9)의 array형태로 총 1647개가 있는 것

sequence_data_v = np.array(X), np.array(y), *[i[1] for i in additional_column_items]
# i[1]로 해둔 이유는 딕셔너리 item에서 value값을 가져오기 위함
# *은 가변인자를 의미함, 어떤 개수의 인자라도 모두 받아서 처리할 때 사용
len(sequence_data_v) # 7
# 위에서 np.array(X)를 하나, np.array(y)를 하나, 나머지 뒷부분의 길이가 5 이므로 총 7이 됨
sequence_data_v
# 순서대로 array(X), array(y), date, item, store, store_item_id, yearly_corr이 들어가있음

# split a multivariate sequence into samples
def split_sequences(group_data, n_steps_in, n_steps_out, x_cols, y_cols, additional_columns, step=1, lag_fns=[]):
    # group_data에 7개로 쪼갠 데이터프레임 split_dfs에서 하나씩 들어옴 ex. split_dfs[0]
    # n_steps_in=180, n_steps_out=90
    # x_cols=y_cols=['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos']
    # additional_columns=['store_item_id', 'item', 'store', 'date', 'yearly_corr']
    # step=1, lag_fns=[last_year_lag]
    X, y = list(), list() # X와 y에 빈 리스트 생성
    additional_col_map = defaultdict(list) # 디폴트값은 리스트
    group_data = group_data.sort_values('date') # 데이터프레임을 date를 기준으로 정렬
    
    # 여기서부터 return까지의 설명은 위에 따로 빼둔 코드에 써뒀음
    for i, lag_fn in enumerate(lag_fns):
        group_data[f'lag_{i}'] = lag_fn(group_data[y_cols[0]])
    
    steps = list(range(0, len(group_data), step))
    if step != 1 and steps[-1] != (len(group_data) - 1):
        steps.append((len(group_data) - 1))
    # if문은 step이 1이기 때문에 걍 무시해도 됨
    for i in steps:
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(group_data):
            break
        # gather input and output parts of the pattern
        if len(x_cols) == 1:
            x_cols = x_cols[0]
        seq_x, seq_y = group_data.iloc[i:end_ix, :][x_cols].values, group_data.iloc[end_ix:out_end_ix, :][y_cols + [f'lag_{i}' for i in range(len(lag_fns))]].values
        for col in additional_columns:
            additional_col_map[col].append(group_data.iloc[end_ix][col])
        X.append(seq_x)
        y.append(seq_y)
    additional_column_items = sorted(additional_col_map.items(), key=lambda x: x[0])
    return (np.array(X), np.array(y), *[i[1] for i in additional_column_items])


def _apply_df(args):
    df, func, key_column = args
    # 아래 split_dfs를 통해 만든 7개의 데이터프레임을 하나씩 가져와서 돌림
    # func=sequence_fn, key_column='store_item_id'
    result = df.groupby(key_column).progress_apply(func)
    # 하나의 데이터프레임을 'store_item_id를 기준으로 그룹화 한 후
    # sequence_fn을 데이터프레임의 모든 열을 기준으로 적용해줌
    # 현재 sequence_fn은 split_sequences임
    # 나중에 diff가 True가 되면 split_sequence_difference로 바뀜
    # 이 때 진행도를 보여주기 위해서 apply대신 progress_apply를 사용
    return result

result = split_dfs[0].groupby('store_item_id').progress_apply(sequence_fn)
# 여기서의 split_dfs는 데이터프레임 7개가 들어있는 리스트, split_dfs[0]가 데이터프레임
# sequence_fn은 split_sequences
# result 하나는 길이 83의 시리즈
# result[0]는 길이 7의 tuple
# result[0][0]은 길이 1647의 array
result[0][0].shape # 1647, 180, 8


def almost_equal_split(seq, num):
    # (scaled_data['store_item_id'].unique(), workers)
    avg = len(seq) / float(num)
    # avg = 500 / 6.0 = 83.33333
    out = []
    last = 0.0
    while last < len(seq): # start : 0 < 500 -> True
        out.append(seq[int(last):int(last + avg)])
        # 83개, 83개, 84개, 83개, 83개, 83개, 1개
        # floating point error로 인해서 마지막에 1개가 떨어져 나가지만 어차피 나중에 다시 concat하기 때문에 상관없음 
        last += avg # 83.33, 166.66, 250, 333.33, 416.66, 500
    return out
# 즉 scaled_data['store_item_id'].unique()의 값들을
# 각각 83, 83, 84, 83, 83, 84개씩 끊어서 리스트를 만듦
# # 위 작업은 결국 성능을 높이기 위한 작업이었음


# 데이터프레임 어떻게 만들어지는지 직접 확인
key_splits = almost_equal_split(scaled_data['store_item_id'].unique(), 6)
key_column = 'store_item_id'
x_cols=['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos']
y_cols=['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos']
additional_columns=['item', 'store', 'date', 'yearly_corr']
df = scaled_data[list(set([key_column] + x_cols + y_cols + additional_columns))]
split_dfs = [df[df[key_column].isin(key_list)] for key_list in key_splits]


sequence_fn = partial(
            split_sequences,
            n_steps_in=180, # 180
            n_steps_out=90, # 90
            x_cols=x_cols, # ['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos']
            y_cols=y_cols, # ['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos']
            additional_columns=list(set([key_column] + additional_columns)),
            # -> ['item', 'date', 'store_item_id', 'yearly_corr', 'store']
            lag_fns=[last_year_lag],
            step=1 # 1
        )


def mp_apply(df, func, key_column):
    # df = scaled_data[list(set([key_column] + x_cols + y_cols + additional_columns))]
    # func = sequence_fn
    # key_column = 'store_item_id'
    workers = 6 # 이 사람 환경에서는 멀티 프로세싱 6개가 가장 효율이 좋았던 것으로 보임
    # pool = mp.Pool(processes=workers)
    key_splits = almost_equal_split(df[key_column].unique(), workers)
    # 위에서 만든 리스트값이 들어가 있음
    split_dfs = [df[df[key_column].isin(key_list)] for key_list in key_splits]
    # 위에서 83개 또는 84개로 store_item_id를 자른것을 기준으로 데이터프레임을 자름
    # 데이터프레임이 총 7개로 잘림 (원래는 6개여야 하지만 floating point error로 인해서 7개가 됨)
    result = process_map(_apply_df, [(d, func, key_column) for d in split_dfs], max_workers=workers)
    # _apply_df_라는 함수에 d, func=sequence_fn, key_column='store_item_id'를 넣어줌
    # d에는 split_dfs에서 만든 7개의 데이터프레임을 하나씩 넣어줌
    # 즉 _apply_df를 총 7번을 돌릴거임
    # process_map의 max_workers=6은 생성할 최대 worker의 수를 6으로 설정하겠다는 의미
    # 즉 멀티프로세싱을 6개까지 할 수 있다는 의미
    return pd.concat(result)
    # _apply_df를 통해 나온 데이터프레임을 하나의 데이터프레임으로 합침


def sequence_builder(data, n_steps_in, n_steps_out, key_column, x_cols, y_col, y_cols, additional_columns, diff=False, lag_fns=[], step=1):
    # data=scaled_data, n_steps_in=180, n_steps_out=90, key_column='store_item_id',
    # x_cols=['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos']
    # y_col='sales'
    # y_cols=['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos']
    # additional_columns=['item', 'store', 'date', 'yearly_corr']
    # lag_fns=[last_year_lag], diff=False, step=1
    
    if diff:
        # multiple y_cols not supported yet
        sequence_fn = partial(
            split_sequence_difference,
            n_steps_in=n_steps_in,
            n_steps_out=n_steps_out,
            x_cols=x_cols,
            y_col=y_col,
            diff=diff,
            additional_columns=list(set([key_column] + additional_columns))
        )
        data['unmod_y'] = data[y_col]
        sequence_data = mp_apply(
            data[list(set([key_column] + x_cols + [y_col, 'unmod_y'] + y_cols + additional_columns))],
            sequence_fn,
            key_column
        )
    else: # 처음에 diff=False 이므로 여기부터 돌아감
        # first entry in y_cols should be the target variable
        # partial은 호출된 function과 그 args, keywords에 대해서 동작하는 객체(함수)를 반환함
        sequence_fn = partial(
            split_sequences,
            n_steps_in=n_steps_in, # 180
            n_steps_out=n_steps_out, # 90
            x_cols=x_cols, # ['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos']
            y_cols=y_cols, # ['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos']
            additional_columns=list(set([key_column] + additional_columns)),
            # -> ['item', 'date', 'store_item_id', 'yearly_corr', 'store']
            lag_fns=lag_fns, # [last_year_lag]
            step=step # 1
        )
        # 
        sequence_data = mp_apply(
            data[list(set([key_column] + x_cols + y_cols + additional_columns))],
            # ['store_item_id, 'sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos', 'item', 'date', 'yearly_corr', 'store']
            # scaled data에서 year, month, day, dayofweek, mean_sales를 없앤것이 현재 data
            # set은 집합이므로 중복되는 열을 없애줌
            sequence_fn, # 위에서 인자들을 새로 정의한 함수
            key_column # 'store_item_id'
        )
        # mp_apply의 결과 split_sequences를 거친 데이터프레임을 다시 하나의 데이터프레임으로 합침
        # 자세한 설명은 사용자 정의 함수들 사이에 있음

    # 이 아래부분이 실행이 안되는데 이해가 안 됨
    # 현재 데이터가 어떻게 정제되었는지는 이해했으니 일단 넘어가기
    sequence_data = pd.DataFrame(sequence_data, columns=['result'])
    s = sequence_data.apply(lambda x: pd.Series(zip(*[col for col in x['result']])), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'result'
    sequence_data = sequence_data.drop('result', axis=1).join(s)
    sequence_data['result'] = pd.Series(sequence_data['result'])
    if diff:
        sequence_data[['x_sequence', 'y_sequence'] + sorted(set([key_column] + additional_columns + ['x_base', 'y_base', 'mean_traffic']))] = pd.DataFrame(sequence_data.result.values.tolist(), index=sequence_data.index)
    # 위에 if문은 무시해도 됨 (diff=False)
    else:
        sequence_data[['x_sequence', 'y_sequence'] + sorted(set([key_column] + additional_columns))] = pd.DataFrame(sequence_data.result.values.tolist(), index=sequence_data.index)
    sequence_data.drop('result', axis=1, inplace=True)
    if key_column in sequence_data.columns:
        sequence_data.drop(key_column, axis=1, inplace=True)
    sequence_data = sequence_data.reset_index()
    print(sequence_data.shape)
    sequence_data = sequence_data[~sequence_data['x_sequence'].isnull()]
    return sequence_data


mp_apply(group_data_1_1, sequence_fn, 'store_item_id')

def last_year_lag(col): 
    return (col.shift(364) * 0.25) + (col.shift(365) * 0.5) + (col.shift(366) * 0.25)

''' 실행 불가
if __name__ == '__main__':
    data = reduce_mem_usage(pd.read_pickle(r'C:\Users\kijeong\research\sales_data/processed_data_test_stdscaler.pkl'))
    sequence_data = sequence_builder(data, 180, 90, 
        'store_item_id', 
        ['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos'], 
        'sales', 
        ['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos'],
        ['item', 'store', 'date', 'yearly_corr'],
        lag_fns=[last_year_lag]
    )
    sequence_data.to_pickle(r'C:\Users\kijeong\research\sales_data/sequence_data_stdscaler_test.pkl')
'''
scaled_data
sequence_data = pd.read_pickle(r'C:\Users\kijeong\research\sales_data/sequence_data_stdscaler_test.pkl')
sequence_data
scaled_data.shape # 958000, 18
sequence_data.shape # 823500, 7
# columns = ['store_item_id', 'x_sequence', 'y_sequence', 'date', 'item', 'store', 'yearly_corr']
# 958000 - 269x500 = 823500
# 위에서 180개 아래에서 90개가 짤리면 총 270개가 짤려서 823000이 되어야 하는거 아닌가..?
sequence_data[['date']].describe() 
# 31+28+31+30+31+29=180이므로 시작이 2013년 6월 30일
# 31+28+31=90이므로 끝이 2018년 1월 1일 (2017년 12월 31일이어야 되는거 아닌가..?)


# 데이터 필터링
lag_null_filter = sequence_data['y_sequence'].apply(lambda val: np.isnan(val[:, -1].reshape(-1)).sum() == 0)
# reshape(-1)은 1차원 array에서 남은 원소로 배열, 즉 -1만 쓰면 차원의 크기 잘 모르니까 니가 알아서 배열해달라는 의미
# sequence_data['y_sequence'].values[0].shape --> (90, 9)
# 'y_sequence'의 경우 값이 들어갈 때 (90, 9)로 들어갔고, 마지막 열에 lag_0에서 nan이 있는 부분이 있었음
# nan이 있으면 True이므로 .sum() == 0이라는 것은 nan이 하나도 없는 부분을 True로 반환함
# 즉 말 그대로 lag에서 계산할 수 없는 nan(null) 값이 있는지 filtering 해달라는 의미

sequence_data.loc[lag_null_filter, 'date'].min() # 위에서 필터링 한 것을 기반으로 nan이 없는 행의 date의 최소를 가져옴
# Timestamp('2014-01-02 00:00:00') --> 왜냐하면 처음에 366개를 밀어서 2013년 전체와 2014년 1월 1일 데이터까지 밀렸기 떄문

# test data로 2018년 1월 1일의 데이터만 가져옴
test_sequence_data = sequence_data[sequence_data['date'] == '2018-01-01']
test_sequence_data



# Prepare pytorch dataloader

class StoreItemDataset(Dataset):
    def __init__(self, cat_columns=[], num_columns=[], embed_vector_size=None, decoder_input=True, ohe_cat_columns=False):
        super().__init__()
        train = None
        self.cat_columns = cat_columns 
        self.num_columns = num_columns
        self.cat_classes = {}
        self.cat_embed_shape = []
        self.cat_embed_vector_size = embed_vector_size if embed_vector_size is not None else {}
        self.pass_decoder_input=decoder_input
        self.ohe_cat_columns = ohe_cat_columns
        self.cat_columns_to_decoder = False

    def get_embedding_shape(self):
        return self.cat_embed_shape

    def load_sequence_data(self, processed_data):
        train = processed_data

    def process_cat_columns(self, column_map=None):
        column_map = column_map if column_map is not None else {}
        for col in self.cat_columns:
            train[col] = train[col].astype('category')
            if col in column_map:
                train[col] = train[col].cat.set_categories(column_map[col]).fillna('#NA#')
            else:
                train[col].cat.add_categories('#NA#', inplace=True)
            self.cat_embed_shape.append((len(train[col].cat.categories), self.cat_embed_vector_size.get(col, 50)))
            # store shape (11,4)
    
    def __len__(self):
        return len(train)

    def __getitem__(self, idx):
        # cat_columns=['store', 'item'], num_columns=['yearly_corr'], embed_vector_size={'store': 4, 'item': 4}, ohe_cat_columns=True
        row = train.iloc[[idx]]
        # idx에 해당하는 인덱스의 행들을 추출해서 row라고 지정
        x_inputs = [torch.tensor(row['x_sequence'].values[0], dtype=torch.float32)]
        # row의 x_sequence열의 값들을 추출해서 첫번째 값(array, shape:180,8)를 tensor로 변환, 이 때 type을 torch.float32로 지정
        y = torch.tensor(row['y_sequence'].values[0], dtype=torch.float32)
        # 이번엔 y_sequence열의 값들을 추출해서 첫번째 값(array, shape:90:9)를 tensor로 변환, 이 때 type을 torch.float32로 지정
        if self.pass_decoder_input:
            decoder_input = torch.tensor(row['y_sequence'].values[0][:, 1:], dtype=torch.float32)
            # decoder_input에는 row['y_sequence'].values[0]의 첫번째 열, 즉 sales를 제거하고 tensor로 변환 
        if len(self.num_columns) > 0:
            for col in self.num_columns:
                num_tensor = torch.tensor([row[col].values[0]], dtype=torch.float32)
                x_inputs[0] = torch.cat((x_inputs[0], num_tensor.repeat(x_inputs[0].size(0)).unsqueeze(1)), axis=1)
                decoder_input = torch.cat((decoder_input, num_tensor.repeat(decoder_input.size(0)).unsqueeze(1)), axis=1)
        if len(self.cat_columns) > 0:
            if self.ohe_cat_columns:
                for ci, (num_classes, _) in enumerate(self.cat_embed_shape):
                    col_tensor = torch.zeros(num_classes, dtype=torch.float32)
                    col_tensor[row[self.cat_columns[ci]].cat.codes.values[0]] = 1.0 # cat.codes.values는 원핫인코딩 비슷(숫자 매핑)
                    col_tensor_x = col_tensor.repeat(x_inputs[0].size(0), 1)
                    x_inputs[0] = torch.cat((x_inputs[0], col_tensor_x), axis=1)
                    if self.pass_decoder_input and self.cat_columns_to_decoder:
                        col_tensor_y = col_tensor.repeat(decoder_input.size(0), 1)
                        decoder_input = torch.cat((decoder_input, col_tensor_y), axis=1)
            else:
                cat_tensor = torch.tensor(
                    [row[col].cat.codes.values[0] for col in self.cat_columns],
                    dtype=torch.long
                )
                x_inputs.append(cat_tensor)
        if self.pass_decoder_input:
            x_inputs.append(decoder_input)
            y = torch.tensor(row['y_sequence'].values[0][:, 0], dtype=torch.float32)
        if len(x_inputs) > 1:
            return tuple(x_inputs), y
        return x_inputs[0], y



# class 분해하기
# __init__
cat_columns = ['store', 'item']
num_columns = ['yearly_corr']
cat_classes = {}
cat_embed_shape = []
cat_embed_vector_size = {'store': 4, 'item': 4}
pass_decoder_input=decoder_input
ohe_cat_columns = True
cat_columns_to_decoder = False
train_sequence_data = sequence_data[(sequence_data['date'] <= '2016-10-01') & (sequence_data['date'] >= '2014-01-02')]
valid_sequence_data = sequence_data[(sequence_data['date'] > '2016-10-01') & (sequence_data['date'] <= '2017-01-01')]

train = train_sequence_data


# process_cat_columns
column_map = {} # column_map=None으로 지정하기 때문
for col in cat_columns:
    train[col] = train[col].astype('category')
    if col in column_map:
        train[col] = train[col].cat.set_categories(column_map[col]).fillna('#NA#')
    # column_map이 비어있으므로 if문 돌아가지 않음
    else:
        train[col].cat.add_categories('#NA#', inplace=True)
    cat_embed_shape.append((len(train[col].cat.categories), cat_embed_vector_size.get(col, 50)))

train[cat_columns[0]].dtype # 현재는 타입이 int8
train[cat_columns[0]] = train[cat_columns[0]].astype('category')
train[cat_columns[0]].dtype # 이제는 타입이 categorical로 변함, CategoricalDtype(categories=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
train[cat_columns[0]].cat.add_categories('#NA#', inplace=True) # 카테고리에 #NA#를 추가
train[cat_columns[0]].dtype
cat_embed_shape.append((len(train[col].cat.categories), cat_embed_vector_size.get(cat_columns[0], 50)))
# cat_embed_shape는 현재 비어있는 리스트
# len(train[col].cat.categories = 11, cat_embed_vector_size.get(cat_columns[0], 50) = 4
# get은 딕셔너리의 메소드로 첫번째 인자로 찾고자 하는 키를 넣고, 두번째 인자로 첫번째 인자에서 넣은 키가 없을 때 넣고자 하는 값을 넣어줌
# cat_embed_vector_size에 store라는 키의 값은 4이므로 4가 됨
# 따라서 cat_embed_shape은 (11,4)가 됨


# data after 10th month will have prediction data in y_sequence
# 현재 모드가 test로 되어있으므로 if문이 돌아감
if mode == 'test':
    train_sequence_data = sequence_data[(sequence_data['date'] <= '2017-10-01') & ((sequence_data['date'] >= '2014-01-02'))]
    valid_sequence_data = pd.DataFrame()
    # 2014년 1월 2일부터 2017년 10월 1일까지의 데이터를 train data로 사용
    # valid data는 일단 비워둠
else:    
    train_sequence_data = sequence_data[(sequence_data['date'] <= '2016-10-01') & (sequence_data['date'] >= '2014-01-02')]
    valid_sequence_data = sequence_data[(sequence_data['date'] > '2016-10-01') & (sequence_data['date'] <= '2017-01-01')]

print(train_sequence_data.shape, valid_sequence_data.shape, test_sequence_data.shape)
# train data는 2014.01.02~2017.10.01 총 684500개의 데이터
# valid data는 비어있고, test data는 2018.01.01 총 500개의 데이터

# StoreItemDataset 클래스 선언
train_dataset = StoreItemDataset(cat_columns=['store', 'item'], num_columns=['yearly_corr'], embed_vector_size={'store': 4, 'item': 4}, ohe_cat_columns=True)
valid_dataset = StoreItemDataset(cat_columns=['store', 'item'], num_columns=['yearly_corr'], embed_vector_size={'store': 4, 'item': 4}, ohe_cat_columns=True)
test_dataset = StoreItemDataset(cat_columns=['store', 'item'], num_columns=['yearly_corr'], embed_vector_size={'store': 4, 'item': 4}, ohe_cat_columns=True)
# 선언한 인자는 각각 위와 같음, 각 dataset이 클래스의 객체가 됨

# 각 데이터를 실행중인 데이터라고 선언해줌
train_dataset.load_sequence_data(train_sequence_data)
valid_dataset.load_sequence_data(valid_sequence_data)
test_dataset.load_sequence_data(test_sequence_data)
# 데이터를 각각 클래스에 넣어서 순차적으로 함수를 적용하면 됨

cat_map = train_dataset.process_cat_columns()
cat_map

if mode == 'valid':
    valid_dataset.process_cat_columns(cat_map)
test_dataset.process_cat_columns(cat_map)

batch_size = 256

train_sequence_data.shape # 684500, 7

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
print(len(train_dataloader), len(valid_dataloader), len(test_dataloader))

for i in test_dataloader:
    print(i)

(X_con, X_dec), y = next(iter(train_dataloader))
X_con.shape, y.shape, X_dec.shape


# Encoder Decoder model
import torch.nn as nn
import torch.optim as optim

'''from ts_models.encoders import RNNEncoder, RNNConcatEncoder, RNNInitEncoder
from ts_models.decoders import DecoderCell, AttentionDecoderCell
from ts_models.encoder_decoder import EncoderDecoderWrapper

from torch_utils.cocob import COCOBBackprop
from torch_utils.trainer import TorchTrainer
import torchcontrib'''

torch.manual_seed(420)
np.random.seed(420)

def smape_loss(y_pred, y_true):
    denominator = (y_true + y_pred) / 200.0
    diff = torch.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return torch.mean(diff)

def smape_exp_loss(y_pred, y_true):
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    y_pred = np.expm1(y_pred)
    y_true = np.expm1(y_true)
    denominator = (y_true + y_pred) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)

def differentiable_smape_loss(y_pred, y_true):
    epsilon = 0.1
    summ = torch.max(torch.abs(y_true) + torch.abs(y_pred) + epsilon, torch.tensor(0.5 + epsilon, device='cuda'))
    smape = torch.abs(y_pred - y_true) / summ
    return torch.mean(smape)

device = 'cuda'

encoder = RNNEncoder(
    input_feature_len=71, 
    rnn_num_layers=1, 
    hidden_size=100,  
    sequence_len=180,
    bidirectional=False,
    device=device,
    rnn_dropout=0.2
)

decoder_cell = DecoderCell(
    input_feature_len=10,
    hidden_size=100,
)

#loss_function = differentiable_smape_loss
#loss_function = differentiable_smape_loss
loss_function = nn.MSELoss()
#loss_function = nn.SmoothL1Loss()
# encoder_optimizer = COCOBBackprop(encoder.parameters(), weight_decay=0)
# decoder_optimizer = COCOBBackprop(decoder_cell.parameters(), weight_decay=0)
# encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=2e-3, weight_decay=1e-)
# decoder_optimizer = torch.optim.AdamW(decoder_cell.parameters(), lr=2e-3, weight_decay=1e-1)

encoder = encoder.to(device)
decoder_cell = decoder_cell.to(device)

model = EncoderDecoderWrapper(
    encoder,
    decoder_cell,
    output_size=90,
    teacher_forcing=0,
    sequence_len=180,
    decoder_input=True,
    device='cuda'
)

model = model.to(device)

encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-2)
decoder_optimizer = torch.optim.AdamW(decoder_cell.parameters(), lr=1e-3, weight_decay=1e-2)

encoder_scheduler = optim.lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=1e-3, steps_per_epoch=len(train_dataloader), epochs=6)
decoder_scheduler = optim.lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=1e-3, steps_per_epoch=len(train_dataloader), epochs=6)

model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
#scheduler = optim.lr_scheduler.OneCycleLR(model_optimizer, max_lr=3e-3, steps_per_epoch=len(train_dataloader), epochs=6)

xb, yb = next(iter(train_dataloader))
xb = [xbi.to(device) for xbi in xb]
yb = yb.to(device)
model(xb, yb).shape

trainer = TorchTrainer(
    'encdec_ohe_std_mse_wd1e-2_do2e-1_test_hs100_tf0_adam',
    model, 
    [encoder_optimizer, decoder_optimizer], 
    loss_function, 
    [encoder_scheduler, decoder_scheduler],
    device, 
    scheduler_batch_step=True,
    pass_y=True,
    #additional_metric_fns={'SMAPE': smape_exp_loss}
)

trainer.lr_find(train_dataloader, model_optimizer, start_lr=1e-5, end_lr=1e-2, num_iter=500)

vd = valid_dataloader if mode == 'valid' else None
trainer.train(6, train_dataloader, vd, resume_only_model=True, resume=True)

vd = valid_dataloader if mode == 'valid' else None
trainer.train(6, train_dataloader, vd, resume_only_model=True, resume=True)

trainer._load_checkpoint(only_model=True)

#trainer._load_best_checkpoint()
if mode == 'valid':
    valid_predictions = trainer.predict(valid_dataloader)

test_predictions = trainer.predict(test_dataloader)

valid_sequence_data.index = range(len(valid_sequence_data))
test_sequence_data.index = range(len(test_sequence_data))

if mode == 'valid':
    valid_sequence_data['predictions'] = pd.Series(valid_predictions.tolist())
test_sequence_data['predictions'] = pd.Series(test_predictions.tolist())

if mode == 'valid':
    valid_sequence_data['X'] = valid_sequence_data['x_sequence'].apply(lambda x: x[:, 0])
    valid_sequence_data['Y'] = valid_sequence_data['y_sequence'].apply(lambda x: x[:, 0])


# Rescale data
def rescale_data(scale_map, data_df, columns=['predictions', 'y_sequence', 'x_sequence']):
    rescaled_data = pd.DataFrame()
    for store_item_id, item_data in tqdm(data_df.groupby('store_item_id', as_index=False)):
        mu = scale_map[store_item_id]['mu']
        sigma = scale_map[store_item_id]['sigma']
        for col in columns:
            item_data[col] = item_data[col].apply(lambda x: (np.array(x) * sigma) + mu)
        rescaled_data = pd.concat([rescaled_data, item_data], ignore_index=True)
    return rescaled_data

if mode == 'valid':
    valid_rescaled = rescale_data(scale_map, valid_sequence_data, columns=['X', 'Y', 'predictions'])
test_rescaled = rescale_data(scale_map, test_sequence_data, columns=['predictions'])

if mode == 'valid':
    valid_sequence_data = valid_rescaled
test_sequence_data = test_rescaled


# Results Analysis
valid_sequence_data.head()

import plotly_express as px
import plotly.graph_objects as go


def plot_sequence_row(row):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(180)), y=row['X'],
                        mode='lines',
                        name='past data'))
    fig.add_trace(go.Scatter(x=list(range(180, 270)), y=row['Y'],
                        mode='lines',
                        name='actual'))
    fig.add_trace(go.Scatter(x=list(range(180, 270)), y=row['predictions'],
                        mode='lines',
                        name='predictions'))
    fig.show()

valid_sequence_data['predictions'] = valid_sequence_data['predictions'].apply(np.array)

def get_col_mean(group_data):
    agg_data = {}
    for col in group_data.columns:
        if col == 'date':
            continue
        agg_data[col] = group_data[col].mean()
    return pd.Series(agg_data)

date_predictions = valid_sequence_data[['X', 'Y', 'predictions', 'date']].groupby('date').apply(get_col_mean)

plot_sequence_row(date_predictions.iloc[-1])

# Get Results
def generate_flat_df(sequence_data, predict_col='predictions', actual_col='Y'):
    flat_df = pd.DataFrame()
    for i, row in sequence_data.iterrows():
        row_df = pd.DataFrame()
        start_date = row['date']
        row_df['date'] = pd.date_range(start_date, periods=90).date.tolist()
        row_df['store'] = row['store']
        row_df['item'] = row['item']
        row_df['predictions'] = row[predict_col]
        if actual_col:
            row_df['sales'] = row[actual_col]
        flat_df = pd.concat([flat_df, row_df], ignore_index=False)
    flat_df.index = range(len(flat_df))
    flat_df['date'] = pd.to_datetime(flat_df['date'])
    return flat_df

if mode == 'valid':
    valid_sequence_data_sample = valid_sequence_data[valid_sequence_data['date'] == '2017-01-01']
    valid_predict_df = generate_flat_df(valid_sequence_data_sample)

test_predict_df = generate_flat_df(test_sequence_data, actual_col=None)

test_predict_df.head()

def smape(y_pred, y_true):
    denominator = (y_true + y_pred) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)

if mode == 'valid':
    smape(valid_predict_df['predictions'], valid_predict_df['sales'])

if mode == 'valid':
    smape(np.round(valid_predict_df['predictions']), valid_predict_df['sales'])


# Plot forecast
item_data = data[data['store_item_id'] == '2_20']
item_forecast = test_predict_df[(test_predict_df['store'] == 2) & (test_predict_df['item'] == 20)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=item_data['date'], y=item_data['sales'],
                    mode='lines',
                    name='past data'))
fig.add_trace(go.Scatter(x=item_forecast['date'], y=item_forecast['predictions'],
                    mode='lines',
                    name='forecast'))
fig.show()


# Get submission file
test_predict_df = test_predict_df.merge(test, on=['date', 'store', 'item'])

test_predict_df.sort_values('id').head()

test_predict_df['predictions'] = np.round(test_predict_df['predictions'])

def save_results(df, name):
    df[['id', 'predictions']].rename(columns={'predictions': 'sales'}).sort_values('id').to_csv(f'./results/{name}.csv', index=False)

save_results(test_predict_df, 'encdec_ohe_std_mse_wd1e-2_do2e-1_test_hs100_tf0_adam_round_day')
 