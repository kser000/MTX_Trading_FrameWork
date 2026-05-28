from argparse import ArgumentParser
import pandas as pd
import mplfinance as mpf
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
START_YEAR = 2021
END_YEAR = 2024



def find_min_max(df):
    """
    Finds and prints the maximum and minimum values for close, volume, RSI, MACD, CCI, and ADX in the dataframe
    up to the contract_month of 202212.

    Args:
        df (pd.DataFrame): The input dataframe with columns including 'contract_month', 'close', 'volume', 
                           'rsi', 'macd', 'cci', and 'adx'.
    """
    index = 0
    for i in range(len(df['contract_month'])):
        if int(df['contract_month'][i]) > 202412:
            break
        else:
            index += 1

    print(f'max close : {np.max(df["close"][:index])}, min close : {np.min(df["close"][:index])}')
    print(f'max volume : {np.max(df["volume"][:index])}, min volume : {np.min(df["volume"][:index])}')
    print(f'max rsi : {np.max(df["rsi"][:index])}, min rsi : {np.min(df["rsi"][:index])}')
    print(f'max macd : {np.max(df["macd"][:index])}, min macd : {np.min(df["macd"][:index])}')
    print(f'max cci : {np.max(df["cci"][:index])}, min cci : {np.min(df["cci"][:index])}')
    print(f'max adx : {np.max(df["adx"][:index])}, min adx : {np.min(df["adx"][:index])}')

def draw_hist(x, name, reverse=False):
    """
    Draws and saves a histogram for the given data.

    Args:
        x (pd.Series or np.ndarray): The data to plot the histogram for.
        name (str): The name to use for the saved histogram image file.
    """
    plt.clf()
    plt.hist(x, bins=30, edgecolor='black', range=(x.min(), x.max()))
    plt.title('Histogram of Random Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    save_dir = 'histogram'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    if reverse == True:
        save_path = os.path.join(save_dir,f'{name}-r.png')
    else:    
        save_path = os.path.join(save_dir,f'{name}.png')
    plt.savefig(save_path, format='png')
    plt.close()

def calculate_ma(arr, length):
    """
    Calculates the Moving Average (MA) for the given data and length.

    Args:
        arr (np.ndarray): The input data array.
        length (int): The length for calculating the MA.

    Returns:
        np.ndarray: The calculated MA values.
    """
    ma = []
    for i in range(len(arr)):
        if i < length-1:
            ma.append(arr[i])
        else:
            ma.append(np.mean(arr[i+1-length:i+1]))
            
    return np.array(ma)

def calculate_ema(arr, window_size):
    """
    Calculates the Exponential Moving Average (EMA) for the given data and window size.

    Args:
        arr (np.ndarray): The input data array.
        window_size (int): The window size for calculating the EMA.

    Returns:
        np.ndarray: The calculated EMA values.
    """
    alpha = 2 / (window_size + 1)
    ema = []
    
    for i in range(len(arr)):
        if i == 0:
            ema.append(arr[i])
        else:
            ema.append(ema[i-1] * (1 - alpha) + arr[i] * alpha)
            
    return np.array(ema)


def add_columns(input_file_path):
    """
    Reads the input CSV file, adds and formats necessary columns.

    Args:
        input_file_path (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: The formatted dataframe.
    """
    df = pd.read_csv(input_file_path)
        
    first_row = df.columns.tolist()
    df.loc[-1] = first_row
    df.index = df.index + 1
    df = df.sort_index()
    df.columns=['stock', 'contract_month', 'date', 'open', 'high', 'low','close','volume']
    df = df.astype({'stock':'str', 'contract_month':'str', 'date':'str', 'open':'float32', 'high':'float32', 'low':'float32','close':'float32','volume':'float32'})
    df['date'] = pd.to_datetime(df['date'])
    return df

def add_MA(df, ma_len):
    """
    Adds a moving average (MA) column to the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing a 'close' column with price data.
        len (int): The window size for calculating the moving average.

    Returns:
        pd.DataFrame: The dataframe with an added column named 'ma{len}', representing the moving average of the 'close' column.

    """
    ma = calculate_ma(df['close'], ma_len)
    df.insert(len(df.columns), f'ma{ma_len}', ma)
    
    return df

def add_EMA(df, ema_len):
    ema = calculate_ema(df['close'], ema_len)
    df.insert(len(df.columns), f'ema{ema_len}', ema)
    return df

def add_trend(df, trend_len, ma):
    """
    Adds the trend column to the dataframe.
    Trend indicates an upward or downward trend.
    
    Args:
        df (pd.DataFrame): The input dataframe with a 'close' and 'ma20' columns.

    Returns:
        pd.DataFrame: The dataframe with the added trend column.
    """   
    # ma method
    # trend = []
    # for i in range(len(df['close'])):
    #     if df.iloc[i]['close'] >= df.iloc[i][f'ma{ma}']:
    #         trend.append(1)
    #     else:
    #         trend.append(-1)
            
    # for i in range(0, len(trend), trend_len):
    #     if sum(trend[i:i+trend_len]) >= 0:
    #         if i+trend_len > len(trend):
    #             trend[i:] = [1] * (len(trend) - i)
    #         else:
    #             trend[i:i+trend_len] = [1] * trend_len
    #         # print(f'index {i} to index {i+trend_len-1} is upward.')
    #     else:
    #         if i+trend_len > len(trend):
    #             trend[i:] = [-1] * (len(trend) - i)
    #         else:
    #             trend[i:i+trend_len] = [-1] * trend_len
    #         # print(f'index {i} to index {i+trend_len-1} is downward.')

    # daily trend method
    trend = [0] * len(df['close'])
    for i in range(0, len(df['close']), trend_len):
        if i+trend_len > len(df['close']):
            if df.iloc[i]['open'] < df.iloc[-1]['close']:
                trend[i:] = [1] * (len(trend) - i)
            else:
                trend[i:] = [-1] * (len(trend) - i)                
        else:
            if df.iloc[i]['open'] < df.iloc[i+trend_len-1]['close']:
                trend[i:i+trend_len] = [1] * trend_len
            else:
                trend[i:i+trend_len] = [-1] * trend_len          
    df.insert(len(df.columns), 'trend', trend) 
    
    return df

def add_MACD(df):
    """
    Adds the MACD column to the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe with a 'close' column.

    Returns:
        pd.DataFrame: The dataframe with the added MACD column.
    """
            
    ema_12 = calculate_ema(df['close'], 12)
    ema_26 = calculate_ema(df['close'], 26)
    macd = ema_12 - ema_26
    
    df.insert(len(df.columns), 'macd', macd)

    return df

def add_RSI(df, n=14):
    """
    Adds the RSI column to the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe with a 'close' column.
        n (int): The period for calculating the RSI.

    Returns:
        pd.DataFrame: The dataframe with the added RSI column.
    """
    up = [0]
    down = [1e-10]
    
    change = 0
    for i in range(1, len(df['close'])):
        change = df['close'][i] - df['close'][i-1]
        if change > 0:
            up.append(change)
            down.append(1e-10)
        elif change < 0:
            up.append(0)
            down.append(abs(change))
        else:
            up.append(0)
            down.append(1e-10)
    
    
    ave_up = calculate_ema(up, n)
    ave_down = calculate_ema(down, n)
    

    rs = ave_up / ave_down
    rsi = 100 - 100 / ( 1 + rs)

    df.insert(len(df.columns), 'rsi', rsi)
    
    return df

def add_MFI(df, n=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    rmf = tp * df['volume']

    direction = tp.diff()
    pos_mf = np.where(direction > 0, rmf, 0.0)
    neg_mf = np.where(direction < 0, rmf, 0.0)

    pos_mf_sum = pd.Series(pos_mf).rolling(n).sum()
    neg_mf_sum = pd.Series(neg_mf).rolling(n).sum()

    mfr = pos_mf_sum / (neg_mf_sum.replace(0, np.nan))
    mfi = 100 - (100 / (1 + mfr))

    df['mfi'] = mfi.fillna(0).values
    return df

def add_CCI(df, n=14):
    """
    Adds the CCI column to the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe with 'high', 'low', and 'close' columns.
        n (int): The period for calculating the CCI.

    Returns:
        pd.DataFrame: The dataframe with the added CCI column.
    """
    tp = []
    md = []
    ma = []
    cci = [0] * 26
    for i in range(len(df['close'])):
        tp.append((df['high'][i] + df['low'][i] + df['close'][i]) / 3)
    
    for i in range(n-1, len(tp)):
        ma.append(np.mean(tp[i-n+1:i+1]))
              
    temp = abs(np.array(tp[n-1:]) - np.array(ma))


    for i in range(n-1, len(temp)):
        md.append(np.mean(temp[i-n+1:i+1]))



    cci = np.append(cci, (np.array(tp[n*2-2:]) - np.array(ma[n-1:])) / (0.015 * np.array(md)))
    cci[np.isnan(cci)] = 0
    df.insert(len(df.columns), 'cci', cci)

    return df

def add_ADX(df,n=14):
    """
    Adds the ADX column to the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe with 'high', 'low', and 'close' columns.
        n (int): The period for calculating the ADX.

    Returns:
        pd.DataFrame: The dataframe with the added ADX column.
    """
    DMP = [1e-10]
    DMM = [1e-10]
    TR = [1e-10]
    for i in range(1, len(df['high'])):
        DMP.append(df['high'][i] - df['high'][i-1] if (df['high'][i] - df['high'][i-1]) > 0 else 0)
        DMM.append(df['low'][i-1] - df['low'][i] if (df['low'][i-1] - df['low'][i]) > 0 else 0)
        TR.append(np.max([df['high'][i] - df['low'][i], 
                          abs(df['high'][i] - df['close'][i-1]),
                          abs(df['low'][i]-df['close'][i-1])]))

    smooth_DMP = calculate_ema(DMP, n)
    smooth_DMM = calculate_ema(DMM, n)
    smooth_TR = calculate_ema(TR, n)
    
    DIP = (smooth_DMP / smooth_TR) * 100
    DIM = (smooth_DMM / smooth_TR) * 100
    
    DX = (abs(DIP - DIM) / (DIP + DIM)) * 100
       
    ADX = calculate_ema(DX, n)
    df.insert(len(df.columns), 'adx', ADX)

    return df

    
def split_csv(df, scalar, save_dir, reverse=False):
    os.makedirs(save_dir, exist_ok=True)
    for cm, g in df.groupby("contract_month"):
        if reverse == True:
            save_path = os.path.join(save_dir, f'mtx-{cm}-{scalar}min-r.csv')
        else:
            save_path = os.path.join(save_dir, f'mtx-{cm}-{scalar}min.csv')
        g = g.iloc[:-1]
        g.to_csv(save_path, index=False)

def reverse_data(df):
    df['open'], df['close'] =  df['close'], df['open']
    df['open'] = df['open'][::-1].values
    df['close'] = df['close'][::-1].values
    df['high'] = df['high'][::-1].values
    df['low'] = df['low'][::-1].values

    return df

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scalar', default=15, type=int, help='The time scalar of market data.')
    parser.add_argument('-r', '--reverse', action='store_true', help='Generate reversal data or not')
    parser.add_argument('--save_dir', default='processed_data', help='The directory to save preprocessed data.')
    parser.add_argument('--train_end', default=2020, type=int)
    parser.add_argument('--train_start', type=int)
    args = parser.parse_args()
    
        
    print(f'start data preprocessing.')
    
    df = pd.read_csv("MTX_rawdata/MTX_15min.csv")
    df.columns = ['date', 'open', 'high', 'low','close','volume', 'contract_month']
    df = df.astype({'date':'str', 'open':'float32', 'high':'float32', 'low':'float32','close':'float32','volume':'float32', 'contract_month':'str'})
    df['date'] = pd.to_datetime(df['date'])
    
    # 移除非交易時間的 K-bar
    print('Removing non-trading hours...')
    print(f'移除前總行數: {len(df)}')
    
    # 定義要移除的時間段
    # 05:00 ~ 08:45 (夜盤收盤到日盤開盤之間)
    # 13:45 ~ 15:00 (日盤收盤到夜盤開盤之間)
    
    # 提取時間部分
    df['time'] = df['date'].dt.time
    
    # 創建移除遮罩
    mask_remove1 = (df['time'] > pd.Timestamp('05:00:00').time()) & (df['time'] < pd.Timestamp('09:00:00').time())
    mask_remove2 = (df['time'] > pd.Timestamp('13:45:00').time()) & (df['time'] < pd.Timestamp('15:15:00').time())
    
    # 移除這些時間段的數據
    df = df[~(mask_remove1 | mask_remove2)].copy()
    df = df.reset_index(drop=True)
    
    print(f'移除了: {mask_remove1.sum() + mask_remove2.sum()} 行非交易時間數據')
    print(f'移除後總行數: {len(df)}')
    
    # 移除臨時的 time 欄位
    df = df.drop('time', axis=1)
    
    if args.reverse:
        df["contract_month"] = df['contract_month'].astype(int)
        df = df.loc[(df['contract_month'] <= args.train_end) & (df['contract_month'] >= args.train_start)]
        df["contract_month"] = df["contract_month"].astype(str)
        df = reverse_data(df)
        df = df.reset_index(drop=True)

    df = add_MACD(df)
    df = add_RSI(df)
    df = add_CCI(df)
    df = add_ADX(df)
    df = add_MFI(df)

    if args.reverse:   
        split_csv(df, args.scalar, save_dir=args.save_dir, reverse=args.reverse)
    else:
        split_csv(df, args.scalar, save_dir=args.save_dir)
        
        
    print(f'Data preprocessing completed.')

    
    


    
