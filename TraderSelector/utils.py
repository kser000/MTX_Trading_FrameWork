"""
Utility functions for time series preprocessing
包含 wavelet denoising 和技術指標計算函數
"""
import pandas as pd
import numpy as np
import pywt
import os


def wavelet_denoising(data, wavelet='db4', level=1):
    """
    Wavelet denoising function (完整序列去噪)
    
    Args:
        data: 輸入數據序列
        wavelet: 小波類型，默認 'db4'
        level: 分解層數，默認 1
        decompose_detail: 是否進一步分解 coeff[1]（高頻部分）
                         True: 將 coeff[1] 分解成兩部分，移除最高頻部分
                         False: 標準去噪
    
    Returns:
        denoised_data: 去噪後的數據
    """
    # Padding 100 個點以減少邊界效應
    padded_data = np.pad(data, 100, mode='edge')

    # 標準去噪流程
    coeff = pywt.wavedec(padded_data, wavelet, mode="per", level=level)
    
    # 估計噪音水平
    sigma = (1 / 0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    
    # 計算通用閾值
    uthresh = sigma * np.sqrt(2 * np.log(len(padded_data)))
    
    # 對細節係數應用軟閾值
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]
    coeff[-level] = np.zeros_like(coeff[-level])
    
    # 重構去噪信號
    denoised_data = pywt.waverec(coeff, wavelet, mode='per')
    
    # 移除 padding
    denoised_data = denoised_data[100:-100]
    
    # 處理長度不匹配的情況
    if len(denoised_data) > len(data):
        denoised_data = denoised_data[:len(data)]
    elif len(denoised_data) < len(data):
        denoised_data = pd.Series(denoised_data).reindex(range(len(data)), method='ffill').values
    
    return denoised_data


def wavelet_denoising_sliding_window(data, window_size=1000, wavelet='db4', level=1):
    """
    滑動窗口去噪：對每個時間點，使用前 window_size 天的數據作為窗口進行去噪
    
    策略：
    - 對於第 i 個時間點，使用 [i-window_size+1, i] 的數據作為窗口
    - 對這個窗口進行 wavelet denoising
    - 只返回窗口最後一個值（當前時間點的去噪值）
    
    Args:
        data: 完整時間序列數據（numpy array）
        window_size: 窗口大小，默認 1000 天
        wavelet: 小波類型，默認 'db4'
        level: 分解層數，默認 1
    
    Returns:
        denoised_data: 去噪後的數據（與輸入長度相同）
    """
    data = np.array(data)
    n = len(data)
    denoised_data = np.zeros_like(data)
    
    # 對於每個時間點
    for i in range(n):
        # 確定窗口範圍：從 max(0, i-window_size+1) 到 i+1
        window_start = max(0, i - window_size + 1)
        window_end = i + 1
        
        # 提取窗口數據
        window_data = data[window_start:window_end]
        
        # 如果窗口大小太小（<50），直接使用原始值
        if len(window_data) < 50:
            denoised_data[i] = data[i]
        else:
            # 對窗口進行去噪
            window_denoised = wavelet_denoising(
                window_data,
                wavelet=wavelet,
                level=level,
            )
            # 只取窗口最後一個值（當前時間點的去噪值）
            denoised_data[i] = window_denoised[-1]
    
    return denoised_data


def wavelet_denoising_open_close_interleaved(df, wavelet='db4', level=1):
    """
    將 open 和 close 穿插合併後去噪，然後再分離
    
    策略：
    1. 將 open 和 close 穿插合併：open[0] close[0] open[1] close[1] ...
    2. 對合併後的序列進行去噪
    3. 去噪後再分成 open_denoised 和 close_denoised
    
    Args:
        df: 包含 open 和 close 數據的 DataFrame
        wavelet: 小波類型，默認 'db4'
        level: 分解層數，默認 1
    
    Returns:
        無返回值，直接修改 DataFrame，添加 open_denoised 和 close_denoised 列
    """
    print(f"  Denoising open and close (interleaved method)...")
    
    # 檢查必要的列是否存在
    if 'open' not in df.columns or 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'open' and 'close' columns")
    
    open_data = df['open'].values
    close_data = df['close'].values
    n = len(open_data)
    
    if len(close_data) != n:
        raise ValueError("open and close must have the same length")
    
    # Step 1: 穿插合併 open 和 close
    # 結果：open[0] close[0] open[1] close[1] ... open[n-1] close[n-1]
    interleaved = np.zeros(2 * n)
    interleaved[0::2] = open_data   # 偶數索引：open
    interleaved[1::2] = close_data   # 奇數索引：close
    
    print(f"    Interleaved sequence length: {len(interleaved)}")
    print(f"    Original open/close length: {n}")
    
    # Step 2: 對合併後的序列進行去噪
    print(f"    Applying wavelet denoising to interleaved sequence...", end=' ', flush=True)
    interleaved_denoised = wavelet_denoising(interleaved, wavelet=wavelet, level=level)
    print(f"done")
    
    # Step 3: 分離去噪後的序列
    # 從 interleaved_denoised 中提取 open_denoised 和 close_denoised
    open_denoised = interleaved_denoised[0::2]  # 偶數索引：open
    close_denoised = interleaved_denoised[1::2]  # 奇數索引：close
    
    # 確保長度匹配
    if len(open_denoised) > n:
        open_denoised = open_denoised[:n]
    if len(close_denoised) > n:
        close_denoised = close_denoised[:n]
    if len(open_denoised) < n:
        open_denoised = np.pad(open_denoised, (0, n - len(open_denoised)), mode='edge')
    if len(close_denoised) < n:
        close_denoised = np.pad(close_denoised, (0, n - len(close_denoised)), mode='edge')
    
    # 添加到 DataFrame
    df['open_denoised'] = open_denoised
    df['close_denoised'] = close_denoised
    
    # 確保數值類型
    df['open_denoised'] = pd.to_numeric(df['open_denoised'], errors='coerce')
    df['close_denoised'] = pd.to_numeric(df['close_denoised'], errors='coerce')
    df['open_denoised'] = df['open_denoised'].fillna(df['open'].iloc[0] if len(df) > 0 else 0)
    df['close_denoised'] = df['close_denoised'].fillna(df['close'].iloc[0] if len(df) > 0 else 0)
    
    print(f"  Denoised open and close using interleaved method")


def wavelet_denoising_ohlc(df, wavelet='db4', level=1):
    """
    分別對 OHLC 進行獨立去噪
    
    策略：
    分別對 open, high, low, close 進行獨立的小波去噪
    
    Args:
        df: 包含 OHLC 數據的 DataFrame
        wavelet: 小波類型，默認 'db4'
        level: 分解層數，默認 1
        decompose_detail: 是否進一步分解 coeff[1]（高頻部分）
                         True: 將 coeff[1] 分解成兩部分，移除最高頻部分
                         False: 標準去噪
    
    Returns:
        無返回值，直接修改 DataFrame，添加去噪列
    """
    # 分別對 OHLC 進行獨立去噪
    ohlc_cols = ['open', 'high', 'low', 'close']
    
    
    for col in ohlc_cols:
        print(f"    Denoising {col}...", end=' ', flush=True)
        data = df[col].values
        

        # 執行去噪
        df[f'{col}_denoised'] = wavelet_denoising(
            data, 
            wavelet=wavelet, 
            level=level,
        )
        print(f"done")
    
    # 確保數值類型
    for col in ['open_denoised', 'high_denoised', 'low_denoised', 'close_denoised']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df['close'].iloc[0] if len(df) > 0 else 0)
    
    print(f"  Denoised OHLC independently")


def wavelet_denoising_ohlc_sliding_window(historical_df, test_df, window_size=1000, wavelet='db4', level=1):
    """
    對 Test 期間的 OHLC 數據進行滑動窗口去噪（避免 look ahead bias）
    
    策略：
    - 使用歷史數據（Train+Val）作為上下文
    - 對 Test 期間的每個點，使用從歷史起點到當前點的數據進行滑動窗口去噪
    - 只使用當前點之前的數據，避免未來信息泄露
    
    Args:
        historical_df: 歷史數據 DataFrame（Train+Val 期間，已去噪或原始數據）
        test_df: Test 期間的 DataFrame（需要去噪）
        window_size: 滑動窗口大小，默認 1000 天
        wavelet: 小波類型，默認 'db4'
        level: 分解層數，默認 1
    
    Returns:
        無返回值，直接修改 test_df，添加去噪列
    """
    # 合併歷史數據和 Test 數據（按時間順序）
    # 注意：這裡只使用原始數據列，不包含去噪列
    ohlc_cols = ['open', 'high', 'low', 'close']
    
    # 提取歷史數據的原始值
    historical_data = {}
    for col in ohlc_cols:
        historical_data[col] = historical_df[col].values
    
    # 對每個 OHLC 列進行滑動窗口去噪
    for col in ohlc_cols:
        print(f"    Denoising {col} (sliding window)...", end=' ', flush=True)
        
        # 合併歷史數據和 Test 數據
        historical_values = historical_data[col]
        test_values = test_df[col].values
        
        # 對 Test 期間的每個點進行去噪
        test_denoised = np.zeros_like(test_values)
        
        for i in range(len(test_values)):
            # 構建從歷史起點到當前 Test 點的完整序列
            # 歷史數據 + Test 數據的前 i+1 個點
            full_sequence = np.concatenate([historical_values, test_values[:i+1]])
            
            # 使用滑動窗口去噪
            # 窗口大小為 min(window_size, len(full_sequence))
            current_window_size = min(window_size, len(full_sequence))
            
            # 提取窗口數據（從序列末尾往前取 window_size 個點）
            window_start = max(0, len(full_sequence) - current_window_size)
            window_data = full_sequence[window_start:]
            
            # 如果窗口大小太小（<50），直接使用原始值
            if len(window_data) < 50:
                test_denoised[i] = test_values[i]
            else:
                # 對窗口進行去噪
                window_denoised = wavelet_denoising(
                    window_data,
                    wavelet=wavelet,
                    level=level,
                )
                # 只取窗口最後一個值（當前 Test 點的去噪值）
                test_denoised[i] = window_denoised[-1]
        
        # 將去噪結果添加到 test_df
        test_df[f'{col}_denoised'] = test_denoised
        print(f"done")
    
    # 確保數值類型
    for col in ['open_denoised', 'high_denoised', 'low_denoised', 'close_denoised']:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        test_df[col] = test_df[col].fillna(test_df['close'].iloc[0] if len(test_df) > 0 else 0)
    
    print(f"  Denoised Test OHLC with sliding window (window_size={window_size})")


def add_K(df, period=14, use_denoised=False):
    """
    計算 KD 指標中的 %K（隨機振盪器的快速線）
    
    Args:
        df: 包含 OHLC 數據的 DataFrame
        period: 計算週期，默認 14
        use_denoised: 是否使用去噪數據，默認 False
                     如果 True，使用 high_denoised, low_denoised, close_denoised
                     如果 False，使用 high, low, close
    
    Returns:
        無返回值，直接修改 DataFrame，添加 'K' 列（或 'K_denoised' 列如果 use_denoised=True）
    """
    
    # 根據 use_denoised 選擇使用的列
    if use_denoised:
        high_col = 'high_denoised'
        low_col = 'low_denoised'
        close_col = 'close_denoised'
        output_col = 'K_denoised'
        
        # 檢查是否有去噪數據
        if high_col not in df.columns or low_col not in df.columns or close_col not in df.columns:
            print(f"    Warning: Denoised columns not found, using raw data instead")
            high_col = 'high'
            low_col = 'low'
            close_col = 'close'
            output_col = 'K'
    else:
        high_col = 'high'
        low_col = 'low'
        close_col = 'close'
        output_col = 'K'
    
    # 計算 N 期內的最高價和最低價
    high_rolling = df[high_col].rolling(window=period, min_periods=1)
    low_rolling = df[low_col].rolling(window=period, min_periods=1)
    
    highest_high = high_rolling.max()
    lowest_low = low_rolling.min()
    
    # 計算 %K = (當前收盤價 - N期內最低價) / (N期內最高價 - N期內最低價) × 100
    numerator = df[close_col] - lowest_low
    denominator = highest_high - lowest_low
    
    # 避免除以零
    df[output_col] = np.where(
        denominator != 0,
        (numerator / denominator) * 100,
        50.0  # 如果最高價等於最低價，設為 50（中性值）
    )
    
    # 確保數值類型
    df[output_col] = pd.to_numeric(df[output_col], errors='coerce')
    df[output_col] = df[output_col].fillna(50.0)


def add_D(df, period=14, smooth_period=3, use_denoised=False):
    """
    計算 KD 指標中的 %D（%K 的移動平均）
    
    Args:
        df: 包含 OHLC 數據的 DataFrame
        period: 計算 %K 的週期，默認 14
        smooth_period: %D 的平滑期數（%K 的移動平均），默認 3
        use_denoised: 是否使用去噪數據，默認 False
                     如果 True，使用 K_denoised 計算 D_denoised
                     如果 False，使用 K 計算 D
    
    Returns:
        無返回值，直接修改 DataFrame，添加 'D' 列（或 'D_denoised' 列如果 use_denoised=True）
    """
    # 根據 use_denoised 選擇使用的列
    if use_denoised:
        k_col = 'K_denoised'
        output_col = 'D_denoised'
    else:
        k_col = 'K'
        output_col = 'D'
    
    # 先計算 %K（如果尚未計算）
    if k_col not in df.columns:
        add_K(df, period=period, use_denoised=use_denoised)
    
    # %D = %K 的 smooth_period 期移動平均
    df[output_col] = df[k_col].rolling(window=smooth_period, min_periods=1).mean()
    
    # 確保數值類型
    df[output_col] = pd.to_numeric(df[output_col], errors='coerce')
    df[output_col] = df[output_col].fillna(50.0)


def add_MA20(df, use_denoised=False):
    """
    計算 20 期移動平均線（MA20）
    
    Args:
        df: 包含價格數據的 DataFrame
        use_denoised: 是否使用去噪數據，默認 False
                     如果 True，使用 close_denoised 計算 MA20_denoised
                     如果 False，使用 close 計算 MA20
    
    Returns:
        無返回值，直接修改 DataFrame，添加 'MA20' 列（或 'MA20_denoised' 列如果 use_denoised=True）
    """
    # 根據 use_denoised 選擇使用的列
    if use_denoised:
        close_col = 'close_denoised'
        output_col = 'MA20_denoised'
        
        # 檢查是否有去噪數據
        if close_col not in df.columns:
            print(f"    Warning: close_denoised column not found, using raw data instead")
            use_denoised = False
    else:
        close_col = 'close'
        output_col = 'MA20'
    
    # 如果 use_denoised 被改為 False，重新設置列名
    if not use_denoised:
        close_col = 'close'
        output_col = 'MA20'
    
    # 計算 20 期簡單移動平均
    df[output_col] = df[close_col].rolling(window=20, min_periods=1).mean()
    
    # 確保數值類型
    df[output_col] = pd.to_numeric(df[output_col], errors='coerce')
    if len(df) > 0:
        fill_value = df[close_col].iloc[0] if not pd.isna(df[close_col].iloc[0]) else 0
    else:
        fill_value = 0
    df[output_col] = df[output_col].fillna(fill_value)


def add_MFI(df, period=14, use_denoised=False):
    """
    計算資金流量指標（Money Flow Index, MFI）
    
    Args:
        df: 包含 OHLCV 數據的 DataFrame
        period: 計算週期，默認 14
        use_denoised: 是否使用去噪數據，默認 False
                     如果 True，使用 high_denoised, low_denoised, close_denoised
                     如果 False，使用 high, low, close
                     volume 始終使用原始數據（不進行去噪）
    
    Returns:
        無返回值，直接修改 DataFrame，添加 'MFI' 列（或 'MFI_denoised' 列如果 use_denoised=True）
    """
    # 檢查是否有 volume 列
    if 'volume' not in df.columns:
        print(f"    Warning: No 'volume' column found, skipping MFI calculation")
        output_col = 'MFI_denoised' if use_denoised else 'MFI'
        df[output_col] = 50.0  # 默認中性值
        return

    # 根據 use_denoised 選擇使用的列
    if use_denoised:
        high_col = 'high_denoised'
        low_col = 'low_denoised'
        close_col = 'close_denoised'
        output_col = 'MFI_denoised'
        
        # 檢查是否有去噪數據
        if high_col not in df.columns or low_col not in df.columns or close_col not in df.columns:
            print(f"    Warning: Denoised columns not found, using raw data instead")
            use_denoised = False
    else:
        high_col = 'high'
        low_col = 'low'
        close_col = 'close'
        output_col = 'MFI'
    
    # 如果 use_denoised 被改為 False，重新設置列名
    if not use_denoised:
        high_col = 'high'
        low_col = 'low'
        close_col = 'close'
        output_col = 'MFI'

    # 計算典型價格 (Typical Price)
    typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
    
    # 計算原始資金流量 (Raw Money Flow)
    # volume 始終使用原始數據
    raw_money_flow = typical_price * df['volume']
    
    # 計算價格變化方向
    price_change = typical_price.diff()
    
    # 正資金流量（價格上漲時）
    positive_flow = raw_money_flow.copy()
    positive_flow[price_change <= 0] = 0
    
    # 負資金流量（價格下跌時）
    negative_flow = raw_money_flow.copy()
    negative_flow[price_change >= 0] = 0
    
    # 計算 period 期內的正負資金流量總和
    positive_sum = positive_flow.rolling(window=period, min_periods=1).sum()
    negative_sum = negative_flow.rolling(window=period, min_periods=1).sum()
    
    # 計算資金流量比率（避免除以零）
    money_ratio = np.where(
        negative_sum != 0,
        positive_sum / negative_sum,
        1.0  # 如果負資金流量為0，設為1.0（表示完全正流量）
    )
    
    # 計算 MFI = 100 - (100 / (1 + Money Ratio))
    df[output_col] = 100 - (100 / (1 + money_ratio))
    
    # 處理極端情況（無窮大或 NaN）
    df[output_col] = np.where(
        np.isinf(df[output_col]) | np.isnan(df[output_col]),
        50.0,  # 默認中性值
        df[output_col]
    )
    
    # 確保數值類型
    df[output_col] = pd.to_numeric(df[output_col], errors='coerce')
    df[output_col] = df[output_col].fillna(50.0)
    
    # 限制範圍在 0-100
    df[output_col] = df[output_col].clip(0, 100)


def add_trend_features(df, threshold=0.001):
    """
    添加趨勢分類，同時計算 trend_raw 和 trend_denoise（只存分類標籤）
    
    策略：
    - trend_raw: 基於原始數據的分類標籤
    - trend_denoise: 基於去噪數據的分類標籤
    - 只存分類標籤（0, 1 或 0, 1, 2），不存百分比值
    
    Args:
        df: 包含 OHLC 數據的 DataFrame
        threshold: 趨勢分類的閾值（百分比，預設 0.1%）
                   如果 threshold = 0，則使用二分類（down=0, up=1）
                   如果 threshold > 0，則使用三分類（down=0, flat=1, up=2）
    
    Returns:
        無返回值，直接修改 DataFrame，添加 'trend_raw', 'trend_denoise' 列
    """
    # 判斷是二分類還是三分類
    is_binary = (threshold == 0.0)
    
    # 檢查是否有去噪數據
    has_denoised = 'close_denoised' in df.columns and 'open_denoised' in df.columns
    
    if is_binary:
        print(f"  Adding trend classification: Binary (threshold=0)")
        print(f"    Down (0): trend <= 0, Up (1): trend > 0")
    else:
        print(f"  Adding trend classification: Three-class (threshold={threshold:.3%})")
    
    # 1. 計算 trend_raw（基於原始數據的分類）
    trend_raw_pct = (df['close'] - df['open']) / df['open']
    trend_raw_pct = pd.to_numeric(trend_raw_pct, errors='coerce').fillna(0)
    
    if is_binary:
        df['trend_raw'] = np.where(trend_raw_pct > 0, 1, 0).astype(int)
    else:
        df['trend_raw'] = np.where(
            trend_raw_pct > threshold, 2,
            np.where(trend_raw_pct < -threshold, 0, 1)
        ).astype(int)
    
    # 2. 計算 trend_denoise（如果有去噪數據）
    if has_denoised:
        trend_denoise_pct = (df['close_denoised'] - df['open_denoised']) / df['open_denoised']
        trend_denoise_pct = pd.to_numeric(trend_denoise_pct, errors='coerce').fillna(0)
        
        if is_binary:
            df['trend_denoise'] = np.where(trend_denoise_pct > 0, 1, 0).astype(int)
        else:
            df['trend_denoise'] = np.where(
                trend_denoise_pct > threshold, 2,
                np.where(trend_denoise_pct < -threshold, 0, 1)
            ).astype(int)
        
        # 計算分類差異計數
        class_diff = (df['trend_raw'] != df['trend_denoise'])
        num_diff = np.sum(class_diff)
        print(f"  Trend diff count: {num_diff} / {len(df)} ({num_diff/len(df)*100:.2f}%)")
        
    else:
        # 沒有去噪數據，設為 NaN
        df['trend_denoise'] = np.nan
        print(f"  No denoised data available, trend_denoise set to NaN")


def add_trend(df, trend_threshold=0.0):
    """
    添加趨勢分類，計算 trend_raw 和 trend_denoise
    
    策略：
    - trend_raw: 基於原始數據的分類標籤
    - trend_denoise: 基於去噪數據的分類標籤
    - 計算方法：(close - 昨天的close) / 昨天的close（百分比變化）
    - 如果 trend_threshold = 0，則使用二分類（down=0, up=1）
    - 如果 trend_threshold > 0，則使用三分類（down=0, flat=1, up=2）
      此時 threshold 是比率閾值（例如 0.001 代表 0.1%）
    
    Args:
        df: 包含 OHLC 數據的 DataFrame，必須包含 'close' 列
           如果存在 'close_denoised' 列，則計算 trend_denoise
        trend_threshold: 趨勢分類的閾值（比率，默認 0.0）
                        如果 threshold = 0，則使用二分類（down=0, up=1）
                        如果 threshold > 0，則使用三分類（down=0, flat=1, up=2）
                        例如：threshold=0.001 表示變化超過 0.1% 才算上漲/下跌
                        例如：threshold=0.01 表示變化超過 1% 才算上漲/下跌
    
    Returns:
        無返回值，直接修改 DataFrame，添加 'trend_raw', 'trend_denoise' 列
    """
    # 判斷是二分類還是三分類
    is_binary = (trend_threshold == 0.0)
    
    # 檢查是否有 close 列
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    if is_binary:
        print(f"  Adding trend classification: Binary (close - previous close)")
        print(f"    Down (0): change <= 0, Up (1): change > 0")
    else:
        print(f"  Adding trend classification: Three-class (threshold={trend_threshold:.4f} = {trend_threshold*100:.2f}%)")
        print(f"    Down (0): change < -{trend_threshold:.4f}, Flat (1): -{trend_threshold:.4f} <= change <= {trend_threshold:.4f}, Up (2): change > {trend_threshold:.4f}")
    
    # 1. 計算 trend_raw（基於原始數據）
    # 計算百分比變化：(close - 昨天的close) / 昨天的close
    close_prev = df['close'].shift(1)
    close_pct_change = (df['close'] - close_prev) / close_prev
    
    # 第一天的值為 NaN，設為 0
    close_pct_change = close_pct_change.fillna(0)
    
    # 根據 threshold 進行分類
    if is_binary:
        # 二分類：下跌=0, 上漲=1
        df['trend_raw'] = np.where(close_pct_change > 0, 1, 0).astype(int)
    else:
        # 三分類：下跌=0, 持平=1, 上漲=2
        df['trend_raw'] = np.where(
            close_pct_change > trend_threshold, 2,
            np.where(close_pct_change < -trend_threshold, 0, 1)
        ).astype(int)
    
    # 確保數值類型
    df['trend_raw'] = pd.to_numeric(df['trend_raw'], errors='coerce').fillna(0).astype(int)
    
    # 2. 計算 trend_denoise（如果有去噪數據）
    if 'close_denoised' in df.columns:
        # 計算百分比變化：(close_denoised - 昨天的close_denoised) / 昨天的close_denoised
        close_denoised_prev = df['close_denoised'].shift(1)
        close_denoised_pct_change = (df['close_denoised'] - close_denoised_prev) / close_denoised_prev
        
        # 第一天的值為 NaN，設為 0
        close_denoised_pct_change = close_denoised_pct_change.fillna(0)
        
        # 根據 threshold 進行分類
        if is_binary:
            # 二分類：下跌=0, 上漲=1
            df['trend_denoise'] = np.where(close_denoised_pct_change > 0, 1, 0).astype(int)
        else:
            # 三分類：下跌=0, 持平=1, 上漲=2
            df['trend_denoise'] = np.where(
                close_denoised_pct_change > trend_threshold, 2,
                np.where(close_denoised_pct_change < -trend_threshold, 0, 1)
            ).astype(int)
        
        # 確保數值類型
        df['trend_denoise'] = pd.to_numeric(df['trend_denoise'], errors='coerce').fillna(0).astype(int)
        
        # 計算分類差異計數
        class_diff = (df['trend_raw'] != df['trend_denoise'])
        num_diff = np.sum(class_diff)
        total = len(df)
        print(f"  Trend diff count: {num_diff} / {total} ({num_diff/total*100:.2f}%)")
        
        # 統計信息
        if is_binary:
            trend_raw_up = (df['trend_raw'] == 1).sum()
            trend_raw_down = (df['trend_raw'] == 0).sum()
            trend_denoise_up = (df['trend_denoise'] == 1).sum()
            trend_denoise_down = (df['trend_denoise'] == 0).sum()
            print(f"  trend_raw: {trend_raw_up} up, {trend_raw_down} down")
            print(f"  trend_denoise: {trend_denoise_up} up, {trend_denoise_down} down")
        else:
            trend_raw_down = (df['trend_raw'] == 0).sum()
            trend_raw_flat = (df['trend_raw'] == 1).sum()
            trend_raw_up = (df['trend_raw'] == 2).sum()
            trend_denoise_down = (df['trend_denoise'] == 0).sum()
            trend_denoise_flat = (df['trend_denoise'] == 1).sum()
            trend_denoise_up = (df['trend_denoise'] == 2).sum()
            print(f"  trend_raw: {trend_raw_down} down, {trend_raw_flat} flat, {trend_raw_up} up")
            print(f"  trend_denoise: {trend_denoise_down} down, {trend_denoise_flat} flat, {trend_denoise_up} up")
    else:
        # 沒有去噪數據，設為 NaN
        df['trend_denoise'] = np.nan
        print(f"  No denoised data available, trend_denoise set to NaN")


def align_denoise_with_trend(df):
    """
    當 trend_raw 和 trend_denoise 不同時，將對應的去噪列改回原始值
    
    策略：
    - 檢查每個時間點的 trend_raw 和 trend_denoise
    - 如果兩者不同，則將該時間點的去噪列（close_denoised, open_denoised, high_denoised, low_denoised）
      改回對應的原始值（close, open, high, low）
    - 這樣可以確保當去噪導致趨勢分類改變時，使用原始數據
    
    Args:
        df: 包含 OHLC 數據和 trend_raw, trend_denoise 的 DataFrame
    
    Returns:
        無返回值，直接修改 DataFrame
    """
    # 檢查必要的列
    if 'trend_raw' not in df.columns or 'trend_denoise' not in df.columns:
        print("  Warning: trend_raw or trend_denoise not found, skipping alignment")
        return
    
    # 找出 trend_raw 和 trend_denoise 不同的位置
    trend_diff = (df['trend_raw'] != df['trend_denoise'])
    num_diff = trend_diff.sum()
    total = len(df)
    
    if num_diff == 0:
        print(f"  No trend differences found, no alignment needed")
        return
    
    print(f"  Found {num_diff} / {total} ({num_diff/total*100:.2f}%) points with trend differences")
    
    # 定義需要對齊的 OHLC 列對
    denoise_pairs = [
        ('close_denoised', 'close'),
        ('open_denoised', 'open'),
        ('high_denoised', 'high'),
        ('low_denoised', 'low')
    ]
    
    # 對每個列對進行對齊
    aligned_count = 0
    for denoise_col, raw_col in denoise_pairs:
        if denoise_col not in df.columns:
            continue
        if raw_col not in df.columns:
            print(f"    Warning: {raw_col} not found, skipping {denoise_col}")
            continue
        
        # 在 trend 不同的位置，將去噪列改回原始值
        df.loc[trend_diff, denoise_col] = df.loc[trend_diff, raw_col]
        aligned_count += 1
    
    print(f"  Aligned {aligned_count} denoised columns with raw values at {num_diff} points")
    