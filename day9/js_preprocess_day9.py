import traceback
import tqdm
import numpy as np
import pandas as pd
from functools import partial
from tqdm.contrib.concurrent import process_map
from collections import defaultdict

tqdm.tqdm().pandas()

#이미 main 코드에서 이해해버린 메모리 감소 함수
#최대한 숫자 부분의 bite를 줄여 메모리 사용을 줄여줌
#감소된 수치를 print하고 감소시킨 df를 return
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
        additional_col_map = defaultdict(list) #디폴트 값이 list인 딕셔너리
        group_data[y_col] = group_data[y_col].diff() 
        # ['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos']에 대해
        # 모든 행에 대하여 바로 전 행의 값을 뺌(시점t 값 - 시점 t-1 값)
        additional_col_map['x_base'] = [] #디폴트에 맞게 딕셔너리 key마다 빈 리스트 삽입
        additional_col_map['y_base'] = []
        additional_col_map['mean_traffic'] = []
        for i in range(diff, len(group_data)): # 0부터 958000까지
            # find the end of this pattern
            x_base = group_data.iloc[i - 1]['unmod_y'] #unmod_y 어디서 튀어나온겨?
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out # i + n_steps_in + n_steps_out
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

# split a multivariate sequence into samples
def split_sequences(group_data, n_steps_in, n_steps_out, x_cols, y_cols, additional_columns, step=1, lag_fns=[]):
    X, y = list(), list()
    additional_col_map = defaultdict(list)
    group_data = group_data.sort_values('date')
    # 이건 원래 설계할 때 lag_fns에 여러 개 들어올 수 있도록 하려 했는데
    # 쓸모가 없었을 뿐 그냥 last_year_lag 적용한다 생각하면 된다.
    for i, lag_fn in enumerate(lag_fns):
        group_data[f'lag_{i}'] = lag_fn(group_data[y_cols[0]])

    
    steps = list(range(0, len(group_data), step))
    if step != 1 and steps[-1] != (len(group_data) - 1):
        steps.append((len(group_data) - 1))
    for i in steps:
        # find the end of this pattern
        # moving window(180, 90)
        # 인코더에 90시점치를 예측하기 위한 180개 + 90개 데이터 생성
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(group_data):
            break
        # gather input and output parts of the pattern
        if len(x_cols) == 1:
            x_cols = x_cols[0]
        seq_x, seq_y = group_data.iloc[i:end_ix, :][x_cols].values, group_data.iloc[end_ix:out_end_ix, :][y_cols + [f'lag_{i}' for i in range(len(lag_fns))]].values
        
    #additional_col 값들 역시 for문을 통해 270시점 짜리로 만듬
        for col in additional_columns:
            additional_col_map[col].append(group_data.iloc[end_ix][col])
        X.append(seq_x)
        y.append(seq_y)
    # 아까 만들었던 additional_col에서 0번째 열 즉, item만 뽑아 한 열로 만들고
    # 총 3 개의 열을 반환
    additional_column_items = sorted(additional_col_map.items(), key=lambda x: x[0])
    return (np.array(X), np.array(y), *[i[1] for i in additional_column_items])


def _apply_df(args):
    #args 즉, 인자로 받은 튜플(데이터, 함수, key_column)을 각각 지정
    df, func, key_column = args

    #df를 먼저 key_column을 기준으로 그룹화한다. 
    #이후 df에 인자로 받았던 func을 적용한다. ->progress_apply는 func의 진행 과정 확인 가능     
    result = df.groupby(key_column).progress_apply(func)
    return result

#결과적으로 sequence를 num의 숫자로 분할하는 작업
#순서대로 진행되어 out에 들어감
def almost_equal_split(seq, num):
    avg = len(seq) / float(num) #시퀀스 길이 / 숫자
    out = [] #결과물이 담길 빈 리스트
    last = 0.0 # avg가 더해질 변수
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def mp_apply(df, func, key_column):
    workers = 6
    # pool = mp.Pool(processes=workers)
#key_column 'store_item_id'열을 6개로 분할하여 key_splits에 저장
    key_splits = almost_equal_split(df[key_column].unique(), workers)
# 6개로 분할된 key_splits에서 순서대로 한 파트씩 뽑음
# key_splits의 분할을 group_data에도 적용하여 df 자체를 분할하고
# split_dfs라는 리스트에 6개의 df를 저장
    split_dfs = [df[df[key_column].isin(key_list)] for key_list in key_splits]
#multi-processing을 위한 process_map 적용
#workers=6으로 설정했으므로 최대 6개의 멀티 프로세스
#6개로 분할되었던 df에 대하여 func을 적용
    result = process_map(_apply_df, [(d, func, key_column) for d in split_dfs], max_workers=workers)
#각각의 df에 멀티프로세싱을 통해 apply가 끝나면 concat으로 다시 합침
#결국 전체 df에 apply한 결과물이 나옴
#앞에서 np.array(X), np.array(y), item열 concat해서 sequence data 생성
    return pd.concat(result)

def sequence_builder(data, n_steps_in, n_steps_out, key_column, x_cols, y_col, y_cols, additional_columns, diff=False, lag_fns=[], step=1):
    if diff: #diff가 true면 이 결과를 따름
        # multiple y_cols not supported yet
        sequence_fn = partial( # partial은 함수의 변형을 여러 번 수행할 때 용이
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
    else: # diff가 false일때, 즉 첫 번째 상황에서는 이 함수를 따름
        # first entry in y_cols should be the target variable
        
    #위에 있던 split_sequences의 인자와 default를 살짝 변경
    # 변경사항은 다음과 같다.
    #1. additional_coulmns에 key_column을 추가
    #2. lag_fns를 아래서 정의한 함수 이용
    #3. step을 default값인 1에서 지정 값으로 변경
        sequence_fn = partial(
            split_sequences,
            n_steps_in=n_steps_in,
            n_steps_out=n_steps_out,
            x_cols=x_cols,
            y_cols=y_cols,
            additional_columns=list(set([key_column] + additional_columns)),#key_column인 store_item_id 삽입
            lag_fns=lag_fns,
            step=step
        )
        sequence_data = mp_apply(
            data[list(set([key_column] + x_cols + y_cols + additional_columns))],
            sequence_fn,
            key_column
        )
    #만들어진 x,y,item을 한 열로 concat했었잖아
    #column 이름을 result로 하는 df로 바꾸자
    sequence_data = pd.DataFrame(sequence_data, columns=['result'])
    s = sequence_data.apply(lambda x: pd.Series(zip(*[col for col in x['result']])), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'result'
    sequence_data = sequence_data.drop('result', axis=1).join(s)
    sequence_data['result'] = pd.Series(sequence_data['result'])
    if diff:
        sequence_data[['x_sequence', 'y_sequence'] + sorted(set([key_column] + additional_columns + ['x_base', 'y_base', 'mean_traffic']))] = pd.DataFrame(sequence_data.result.values.tolist(), index=sequence_data.index)
    else:
        sequence_data[['x_sequence', 'y_sequence'] + sorted(set([key_column] + additional_columns))] = pd.DataFrame(sequence_data.result.values.tolist(), index=sequence_data.index)
    sequence_data.drop('result', axis=1, inplace=True)
    if key_column in sequence_data.columns:
        sequence_data.drop(key_column, axis=1, inplace=True)
    sequence_data = sequence_data.reset_index()
    print(sequence_data.shape)
    sequence_data = sequence_data[~sequence_data['x_sequence'].isnull()]
    return sequence_data

# 나름의 보정? 0.25 * (1+B)^2
def last_year_lag(col): return (col.shift(364) * 0.25) + (col.shift(365) * 0.5) + (col.shift(366) * 0.25)

if __name__ == '__main__':
    data = reduce_mem_usage(pd.read_pickle(r'C:\winter\my_practice\day9\data\processed_data_test_stdscaler.pkl'))
    sequence_data = sequence_builder(data, 180, 90, 
        'store_item_id', 
        ['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos'], 
        'sales', 
        ['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos'],
        ['item', 'store', 'date', 'yearly_corr'],
        lag_fns=[last_year_lag]
    )
    sequence_data.to_pickle(r'C:\winter\my_practice\day9\sequence_data\sequence_data_stdscaler_test.pkl')