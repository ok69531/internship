
%load_ext autoreload
%autoreload 2

import torch
import numpy as np
import pandas as pd
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm
tqdm.pandas()

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (12.0, 12.0)



train = pd.read_csv(r'C:\winter\data\demand-forecasting-kernels-only\train.csv')
test = pd.read_csv(r'C:\winter\data\demand-forecasting-kernels-only\test.csv')

#sas에서 univariate처럼 기초통계량 제공
train.describe()

#데이터 확인
print(train.shape)
train.head()

print(test.shape)
test.head()

#8일차에서 feature 생성할 때처럼 date를 먼저 to_datetime으로 변환
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

print(train['date'].min(), train['date'].max()) # 최대값, 최소값 확인 -> 즉, 시작 날짜와 끝 날짜 확인
print(test['date'].min(), test['date'].max())

test['sales'] = np.nan
data = pd.concat([train, test], ignore_index=True) # test를 train 밑으로 붙임 (test가 마지막 5개월치 데이터니까)
data['store_item_id'] = data['store'].astype(str) + '_' + data['item'].astype(str) #매장과 품목을 합쳐서 확인 가능

data['store_item_id'].describe() #store_item 잘 생성되었나 확인

# 요일, 월, 년도, 일 특성을 제작
data['dayofweek'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['day'] = data['date'].dt.day

# month, year, day와 달리 2000대의 연도에 대한 정규화
# min-max 정규화 사용
# (data - min)/(max-min)
data['year'].min(), data['year'].max() 
data['year_mod'] = (data['year'] - data['year'].min()) / (data['year'].max() - data['year'].min())

data.head()

#8일차 특성 생성처럼 sin,cos값 생성하기 위한 함수 정의
#set은 집합으로 만들어 unique시켜주는 테크닉
#list의 중복 없는 길이 구할 때 유용
#sin, cos를 통해 주기를 학습
#길이로 나누는 것은 주기를 파악하기 위함
def sin_transform(values):
    return np.sin(2*np.pi*values/len(set(values)))

def cos_transform(values):
    return np.cos(2*np.pi*values/len(set(values)))

# day, month, dayofweek에 대해 sin,cos 변환
# 주기가 있는 세 변수랑 달리 year는 돌아오지 않으므로 생성 x
data['dayofweek_sin'] = sin_transform(data['dayofweek'])
data['dayofweek_cos'] = cos_transform(data['dayofweek'])
data['month_sin'] = sin_transform(data['month'])
data['month_cos'] = cos_transform(data['month'])
data['day_sin'] = sin_transform(data['day'])
data['day_cos'] = cos_transform(data['day'])

# 아까 nan으로 돌린 id는 쓸모없으니까 버려주고
data.drop('id', axis = 1, inplace=True)

# 매장_품목별로 나눠진 시계열 데이터
data = data.sort_values(['store_item_id', 'date'])


#ACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import plotly.express as px
import plotly.graph_objects as go

def plot_line(acf):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(acf))), y=acf, # import한 acf로 바로 구해서 플롯
    mode='lines',
    name='acf'))
    fig.show()

train['store_item_id'] = train['store'].astype(str) + '_' + train['item'].astype(str)

item_data = train[train['store_item_id'] == '1_1'].sort_values('date') #일단 1번매장, 1번 품목에 대해
#혹시 모르니까 날짜로 다시 sort 해줭

avg_data = train[['date', 'sales']].groupby('date').mean()

#366일인 윤년이 껴있어서 모든 년도에 대해 연도별로 자기상관이 존재하는지 확인하기 위해 366 사용
# nlags가 366이면 길이 367인 벡터 ->오늘이랑 366일 뒤의 값을 비교 가능
item_ac = acf(item_data['sales'], nlags=366)
#그 날 모든 매장과 품목에 대한 sales의 평균들이 자기상관 존재하는지 확인
avg_ac = acf(avg_data['sales'], nlags= len(avg_data))

px.line(item_data, x ='date', y='sales')
plot_line(item_ac)
plot_line(avg_ac)

important_lags = np.argsort(-avg_ac)[:100] #avg_ac를 내림차순으로 정렬하는 index를 뽑기 위해 -avg_ac애 arg_sort사용
important_lags
sample_data = data[data['store_item_id'].isin(pd.Series(data['store_item_id'].unique()).sample(10))]
px.line(sample_data, x ='date', y='sales', color='store_item_id', title='Store item sales')

# 데이터 정규화
train['store_item_id'] = train['store'].astype('str') + '_' + train['item'].astype(str)

# val과 test 때의 데이터는 서로 다르니까
#mode랑 if로 구분해둔겨
#mode = 'valid'
mode = 'test'

if mode == 'valid':
    scale_data = train[train['date'] < '2017-01-01']
else:
    scale_data = train[train['date'] >= '2014-01-01']

#Get yearly autocorrelation for each timeseries

def get_yearly_autocorr(data):
    ac = acf(data, nlags=366)
    return (0.5 * ac[365]) + (0.25 * ac[364]) + (0.25 * ac[366])

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scale_map = {}
scaled_data = pd.DataFrame()

#파이썬의 pickle(피클) 패키지는 list, dict와 같은 파이썬 객체를 그 형태 그대로 저장하고, 불러올 수 있게끔 하는 패키지​
import pickle

def save_scale_map(name, scale_map):
    with open(name + '.pickle', 'wb') as handle: # pickle로 저장할 파일이 쓰기를 위해 binary mode로 열린다는 뜻
        pickle.dump(scale_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #pickle.dump는 쓰기
        #pickle.HIGHEST_PROTOCOL은 대용량 파일 저장할때 메모리가 부족해서 실행이 중단되는 것을 방지

for store_item_id, item_data in tqdm(data.groupby('store_item_id', as_index=False)):
    sidata = scale_data.loc[scale_data['store_item_id'] == store_item_id, 'sales'] #store_item_id별로 sales
    mu = sidata.mean() #각 매장_품목 별 sales의 평균
    sigma = sidata.std() #각 매장_품목 별 sales의 표준편차
    yearly_autocorr = get_yearly_autocorr(sidata) #각 매장_품목 별 sales의 연간 자기상관계수
    item_data.loc[:,'sales'] = (item_data['sales'] - mu) / sigma # mu, sigma 구한걸로 sales를 정규화
    scale_map[store_item_id] = {'mu': mu, 'sigma': sigma} # scale_map dictionary에 "매장_품목" : (평균:mu,표준편차:sigma) 저장
    item_data['mean_sales'] = mu # mu를 열로 추가
    item_data['yearly_corr'] = yearly_autocorr # 연도에 따른 자기상관계수 열로 추가
    scaled_data = pd.concat([scaled_data, item_data], ignore_index=True) #빈 df에 각 매장_품목별 정규화된 데이터 계속해서 concat

# 새로 생긴 자기상관계수 열에 대해 합쳐진 데이터에서 정규화 진행
scaled_data['yearly_corr'] = ((scaled_data['yearly_corr'] - scaled_data['yearly_corr'].mean()) / scaled_data['yearly_corr'].std())
# 새로 생긴 mean_Sales 열에 대해 합쳐진 데이터에서 정규화 진행
scaled_data['mean_sales'] = (scaled_data['mean_sales'] - scaled_data['mean_sales'].mean()) / scaled_data['mean_sales'].std()

scaled_data.loc[scaled_data['store_item_id'] == '1_1', 'sales'].plot() #1_1품목의 정규화된 sales의 시계열도

# memory 사용량 줄여주고 얼마나 감소됐는지 확인할 수 있는 함수
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: # np iinfo : 표현 한계 
                    # 안에 들어있으면
                    df[col] = df[col].astype(np.int8) #np.int8로
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


# 이미 사인,코사인 변환으로 주기까지 표현되어 쓸모없는 원 데이터들은 삭제
scaled_data.drop(['date','day', 'month', 'year', 'dayofweek', 'mean_sales', 'store_item_id'], axis=1).head()
# 메모리 감소 및 확인
scaled_data = reduce_mem_usage(scaled_data)

scaled_data.head()


scaled_data.to_pickle(r'C:\winter\my_practice\day9\data\processed_data_test_stdscaler.pkl')

sequence_data = pd.read_pickle('./sequence_data/sequence_data_stdscaler_test.pkl')

sequence_data.shape


#dataloader

from torch.utils.data import Dataset, DataLoader

class StoreItemDataset(Dataset):
    def __init__(self, cat_columns=[], num_columns=[], embed_vector_size=None, decoder_input=True, ohe_cat_columns=False):
        super().__init__()
        #클래스에서 필요한 변수들을 정의
        self.sequence_data = None
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.cat_classes = {}
        self.cat_embed_shape = []
        #if는 결과물 뒤에 조건으로 사용해도 가능
        self.cat_embed_vector_size = embed_vector_size if embed_vector_size is not None else {}
        self.pass_decoder_input=decoder_input
        self.ohe_cat_columns = ohe_cat_columns
        self.cat_columns_to_decoder = False

    def get_embedding_shape(self):
        # 일단은 빈 리스트 리턴
        return self.cat_embed_shape

    def load_sequence_data(self, processed_data):
        #None이었던 self.sequence_data에 processed_data라는 입력값 넣어줌
        self.sequence_data = processed_data

    def process_cat_columns(self, column_map=None):
        #default가 None이므로 일단 빈 딕셔너리, 다른 값 넣어주면 그 값
        column_map = column_map if column_map is not None else {}
        for col in self.cat_columns:
            #self.sequnce_data의 col을 범주형으로 변경한다.
            self.sequence_data[col] = self.sequence_data[col].astype('category')
            if col in column_map:
                #col이 column_map에 들어있으면 column_map[col]의 값들을 새로운 카테고리로 추가한다.
                # 결측값이 존재하면 '#NA#'라는 값을 대신 삽입한ㄷ다.
                self.sequence_data[col] = self.sequence_data[col].cat.set_categories(column_map[col]).fillna('#NA#')
            else:
                # col이 column_map에 없으면 그냥 #NA#라는 카테고리만 sequence_data의 범주에 추가한다.
                self.sequence_data[col].cat.add_categories('#NA#', inplace=True)
            self.cat_embed_shape.append((len(self.sequence_data[col].cat.categories), self.cat_embed_vector_size.get(col, 50)))
    
    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        row = self.sequence_data.iloc[[idx]]
        x_inputs = [torch.tensor(row['x_sequence'].values[0], dtype=torch.float32)]
        y = torch.tensor(row['y_sequence'].values[0], dtype=torch.float32)
        if self.pass_decoder_input:
            decoder_input = torch.tensor(row['y_sequence'].values[0][:, 1:], dtype=torch.float32)
        if len(self.num_columns) > 0:
            for col in self.num_columns:
                num_tensor = torch.tensor([row[col].values[0]], dtype=torch.float32)
                x_inputs[0] = torch.cat((x_inputs[0], num_tensor.repeat(x_inputs[0].size(0)).unsqueeze(1)), axis=1)
                decoder_input = torch.cat((decoder_input, num_tensor.repeat(decoder_input.size(0)).unsqueeze(1)), axis=1)
        if len(self.cat_columns) > 0:
            if self.ohe_cat_columns:
                for ci, (num_classes, _) in enumerate(self.cat_embed_shape):
                    col_tensor = torch.zeros(num_classes, dtype=torch.float32)
                    col_tensor[row[self.cat_columns[ci]].cat.codes.values[0]] = 1.0
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



sequence_data.head()


lag_null_filter = sequence_data['y_sequence'].apply(lambda val: np.isnan(val[:, -1].reshape(-1)).sum() == 0)

sequence_data.loc[lag_null_filter, 'date'].min()



#Validation scheme
#Validation Model is trained on 2014 to 2016 data and predict 1st 3 months of 2017 data.
#The best performing model will be trained on 2014 to 2017 data to predict first 3 months of 2018, without validation
#In the validation model, sequences of the last 3 months of 2016 shouldn't be included because it contains 2017 values in y

test_sequence_data = sequence_data[sequence_data['date'] == '2018-01-01']

# data after 10th month will have prediction data in y_sequence
if mode == 'test':
    train_sequence_data = sequence_data[(sequence_data['date'] <= '2017-10-01') & ((sequence_data['date'] >= '2014-01-02'))]
    valid_sequence_data = pd.DataFrame()
else:    
    train_sequence_data = sequence_data[(sequence_data['date'] <= '2016-10-01') & (sequence_data['date'] >= '2014-01-02')]
    valid_sequence_data = sequence_data[(sequence_data['date'] > '2016-10-01') & (sequence_data['date'] <= '2017-01-01')]

print(train_sequence_data.shape, valid_sequence_data.shape, test_sequence_data.shape)

train_dataset = StoreItemDataset(cat_columns=['store', 'item'], num_columns=['yearly_corr'], embed_vector_size={'store': 4, 'item': 4}, ohe_cat_columns=True)
valid_dataset = StoreItemDataset(cat_columns=['store', 'item'], num_columns=['yearly_corr'], embed_vector_size={'store': 4, 'item': 4}, ohe_cat_columns=True)
test_dataset = StoreItemDataset(cat_columns=['store', 'item'], num_columns=['yearly_corr'], embed_vector_size={'store': 4, 'item': 4}, ohe_cat_columns=True)

train_dataset.load_sequence_data(train_sequence_data)
valid_dataset.load_sequence_data(valid_sequence_data)
test_dataset.load_sequence_data(test_sequence_data)

cat_map = train_dataset.process_cat_columns()

if mode == 'valid':
    valid_dataset.process_cat_columns(cat_map)

test_dataset.process_cat_columns(cat_map)
batch_size = 256

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

print(len(train_dataloader), len(valid_dataloader))

(X_con, X_dec), y = next(iter(train_dataloader))

X_con.shape, y.shape, X_dec.shape

#Encoder Decoder model
import torch.nn as nn
import torch.optim as optim

# 얘가 직접 짠 모델들임
from ts_models.encoders import RNNEncoder, RNNConcatEncoder, RNNInitEncoder
from ts_models.decoders import DecoderCell, AttentionDecoderCell
from ts_models.encoder_decoder import EncoderDecoderWrapper
from ts_models.encoder_decoder_attention import EncoderDecoderAttentionWrapper

from torch_utils.cocob import COCOBBackprop
from torch_utils.trainer import TorchTrainer
import torchcontrib

torch.manual_seed(420)
np.random.seed(420)


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