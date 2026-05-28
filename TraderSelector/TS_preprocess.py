"""
Time Series Preprocessing - Paper Style
按照論文方法對 Train+Val 期間做完整去噪
輸出可直接用於訓練的 CSV 文件

Usage:
    python TS_preprocess.py --config TraderSelector/config/2025H1.yaml
"""
import pandas as pd
import numpy as np
import yaml
from argparse import ArgumentParser
from utils import *




def preprocess_train_val(config_path, raw_data_path='MTX_rawdata/MTX_daily.csv', trend_threshold=0.001, output_dir='8D_binary'):

    print("="*80)
    print("TIME SERIES PREPROCESSING (Paper Style)")
    print("="*80)
    
    # 讀取配置
    print(f"\nStep 1: Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_start = config['dataset']['train']['start_date']
    train_end = config['dataset']['train']['end_date']
    val_start = config['dataset']['val']['start_date']
    val_end = config['dataset']['val']['end_date']
    test_start = config['dataset']['test']['start_date']
    test_end = config['dataset']['test']['end_date']
    
    print(f"  Train: {train_start} to {train_end}")
    print(f"  Val:   {val_start} to {val_end}")
    print(f"  Test:  {test_start} to {test_end}")
    
    # 讀取原始數據
    print(f"\nStep 2: Loading raw data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 確保 OHLC 列是數值類型
    print(f"  Ensuring OHLC data types...")
    ohlc_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlc_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if pd.isna(df[col]).any():
                print(f"    Warning: Found NaN values in {col}, filling with 0")
                df[col] = df[col].fillna(0)
    
    print(f"  Total data: {len(df)} days")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # 一律對 Train+Val 期間做去噪（確保訓練和驗證使用相同分佈）
    denoise_start = train_start
    denoise_end = val_end  # 改為 val_end，包含整個 Train+Val 期間
    denoise_df = df[(df['date'] >= denoise_start) & (df['date'] <= denoise_end)].copy()
    
    print(f"\nStep 3: Denoising Train+Val period")
    print(f"  Period: {denoise_start} to {denoise_end}")
    print(f"  Samples: {len(denoise_df)} days")
    
    # 對 OHLC 做完整序列去噪（分別獨立去噪）
    print(f"\nStep 4: Wavelet denoising (OHLC independently)")
    print(f"  Denoising each OHLC column independently...")
    
    # 準備保存轉換矩陣的目錄
    config_name = config_path.split('/')[-1].replace('.yaml', '')
    
    # 使用獨立去噪函數（分別對 OHLC 去噪），並保存轉換矩陣
    wavelet_denoising_ohlc(denoise_df, wavelet='db4', level=1)
    #wavelet_denoising_open_close_interleaved(denoise_df, wavelet='db4', level=1)
    # 統計去噪效果
    print(f"  Denoising statistics:")
    for col in ['open', 'close']:
        original_values = denoise_df[col].values
        denoised_values = denoise_df[f'{col}_denoised'].values
        noise = original_values - denoised_values
        noise_std = np.std(noise)
        print(f"    {col}: noise std = {noise_std:.2f}")
    
    # 提取 Test 期間的數據（使用滑動窗口去噪，避免 look ahead bias）
    print(f"\nStep 6: Adding Test period (sliding window denoising)")
    test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
    print(f"  Test period: {test_start} to {test_end}")
    print(f"  Test samples: {len(test_df)} days")
    
    # 對 Test 數據使用滑動窗口去噪（使用 Train+Val 作為歷史上下文）
    if len(test_df) > 0:
        print(f"  Applying sliding window denoising to Test period...")
        print(f"    Using Train+Val data as historical context...")
        print(f"    Window size: 1000 days")
        print(f"    Strategy: For each test point, use data from historical start to current point only")
        
        # 使用滑動窗口去噪，將 Train+Val 數據作為歷史上下文
        wavelet_denoising_ohlc_sliding_window(
            historical_df=denoise_df,  # Train+Val 期間的數據（已去噪或原始數據都可以，這裡用原始數據）
            test_df=test_df,
            window_size=1000,
            wavelet='db4',
            level=1
        )
        print(f"  done")
        
        

    
    # 合併 Train+Val (完整去噪) 和 Test (denoise = raw)
    print(f"\nStep 7: Combining Train+Val (full denoised) and Test (denoise = raw)")
    combined_df = pd.concat([denoise_df, test_df], ignore_index=True)
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    
    #add_K(combined_df)
    #add_D(combined_df)
    #add_MA20(combined_df)
    #add_MFI(combined_df)

    add_trend_features(combined_df, threshold=trend_threshold)
    #align_denoise_with_trend(combined_df)
    #add_trend(combined_df)
    # 確保所有數值列都是正確的數據類型
    print(f"  Ensuring data types...")
    
    # 確保 OHLC 列是數值類型
    ohlc_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlc_columns:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            if pd.isna(combined_df[col]).any():
                print(f"    Warning: Found NaN values in {col}, filling with 0")
                combined_df[col] = combined_df[col].fillna(0)
    
    # 確保去噪列是數值類型
    denoised_columns = ['open_denoised', 'high_denoised', 'low_denoised', 'close_denoised']
    for col in denoised_columns:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            if pd.isna(combined_df[col]).any():
                print(f"    Warning: Found NaN values in {col}, filling with 0")
                combined_df[col] = combined_df[col].fillna(0)
    
    # 確保趨勢分類列是正確的數據類型
    # 確保 trend_raw 和 trend_denoise 的數據類型
    for col in ['trend_raw', 'trend_denoise']:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            if pd.isna(combined_df[col]).any():
                print(f"    Warning: Found NaN values in {col}, filling with 0")
                combined_df[col] = combined_df[col].fillna(0)
            combined_df[col] = combined_df[col].astype(int)
    
    # 確保技術指標列是正確的數據類型
    indicator_columns = ['K', 'D', 'MA20', 'MFI']
    for col in indicator_columns:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            if pd.isna(combined_df[col]).any():
                print(f"    Warning: Found NaN values in {col}, filling with default")
                # 根據指標類型填充默認值
                if col in ['K', 'D', 'MFI']:
                    combined_df[col] = combined_df[col].fillna(50.0)  # 50 是 KD 和 MFI 的中性值
                else:  # MA20
                    combined_df[col] = combined_df[col].fillna(combined_df['close'].iloc[0] if len(combined_df) > 0 else 0)
    
    print(f"  Combined samples: {len(combined_df)} days")
    print(f"    Train+Val (full denoised): {len(denoise_df)} days")
    print(f"    Test (denoise = raw): {len(test_df)} days")
    
    # 保存合併後的數據
    output_file = f"TradeSelector_processed_data/{output_dir}/{config_name}.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nStep 8: Saving combined data")
    print(f"  Output: {output_file}")
    print(f"    Rows: {len(combined_df)}")
    print(f"    Columns: {len(combined_df.columns)}")
    
    # 統計摘要
    
    print(f"\nProcessing strategy:")
    print(f"  ✅ Train+Val 期間: 完整序列去噪處理")
    print(f"  ✅ Test 期間: 滑動窗口去噪（避免 look ahead bias，使用 Train+Val 作為歷史上下文）")
    print(f"  📊 去噪樣本數: {len(denoise_df)} 天 (Train+Val)")
    print(f"  📊 滑動窗口去噪樣本數: {len(test_df)} 天 (Test)")
    print(f"  📊 總樣本數: {len(combined_df)} 天")
    
    print(f"\nPrice ranges (Train+Val denoised):")
    for col in ['open', 'high', 'low', 'close']:
        denoised_col = f'{col}_denoised'
        if denoised_col in denoise_df.columns:
            print(f"  {denoised_col:20s}: [{denoise_df[denoised_col].min():8.2f}, {denoise_df[denoised_col].max():8.2f}]")
    
    
    print(f"\n4. File structure:")
    print(f"   → 前 {len(denoise_df)} 行: Train+Val (去噪)")
    print(f"   → 後 {len(test_df)} 行: Test (原始)")
    print(f"   → 按日期順序排列")
    
    print(f"\n{'='*80}")
    
    return output_file


if __name__ == "__main__":
    parser = ArgumentParser(description='Preprocess time series data (paper style)')
    parser.add_argument('--config', type=str, required=True, 
                        help='Config file path (e.g., TraderSelector/config/2025H1.yaml)')
    parser.add_argument('--raw_data', type=str, 
                        default='MTX_rawdata/MTX_daily.csv',
                        help='Raw data file path')
    parser.add_argument('--trend_threshold', type=float, default=0.0,
                        help='Threshold for trend classification (default: 0.1% as decimal)')
    parser.add_argument('--output_dir', type=str, default='8D_binary',
                        help='Output directory')
    args = parser.parse_args()
    
    # 執行預處理：一律對 Train+Val 去噪，Test 保持原始
    output_file = preprocess_train_val(args.config, args.raw_data, args.trend_threshold, args.output_dir)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"📁 Output file: {output_file}")
    print(f"   Strategy: Train+Val 去噪 + Test 原始，合併輸出")
    print(f"   Trend threshold: {args.trend_threshold:.3%} (percentage)")

