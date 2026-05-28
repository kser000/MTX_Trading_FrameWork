# -*- coding: utf-8 -*-
"""
依 separate_model_spec.md 將 raw 處理成一行一盤格式，並加入 log_return、rolling_std、技術指標（MACD/MFI/CCI/ADX）、n225/sp500/twse、type（夜盤/日盤）。
輸出欄位: date, stock, contract month, open, high, low, close, volume, log_return, rolling_std, macd, mfi, cci, adx, n225, sp500, twse, type
"""
import os
import glob
import pandas as pd
import numpy as np

# 若從 PatchTST 目錄執行，可正確找到 utils
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from utils import fut_to_one_row_per_session


def _add_macd(df, close_col='close', fast=12, slow=26, signal=9):
    """MACD、signal、histogram，以 close 計算。"""
    c = df[close_col].astype(float)
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd


def _add_mfi(df, high_col='high', low_col='low', close_col='close', volume_col='volume', period=14):
    """Money Flow Index，需 high/low/close/volume。"""
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    vol = df[volume_col].astype(float)
    typical = (high + low + close) / 3.0
    raw_mf = typical * vol
    prev_typical = typical.shift(1)
    pos = raw_mf.where(typical > prev_typical, 0.0).rolling(window=period).sum()
    neg = raw_mf.where(typical < prev_typical, 0.0).rolling(window=period).sum()
    ratio = np.where(neg != 0, pos / neg, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + ratio)
    mfi = np.where(np.isfinite(mfi), mfi, 50.0)
    return pd.Series(mfi, index=df.index)


def _add_cci(df, high_col='high', low_col='low', close_col='close', period=20):
    """Commodity Channel Index，typical price = (H+L+C)/3。"""
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    typical = (high + low + close) / 3.0
    sma_tp = typical.rolling(window=period).mean()
    mean_dev = typical.rolling(window=period).apply(lambda x: np.abs(x - np.mean(x)).mean(), raw=True)
    cci = (typical - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
    return cci.fillna(0.0)


def _add_adx(df, high_col='high', low_col='low', close_col='close', period=14):
    """ADX（Average Directional Index），需 high/low/close，period 常用 14。"""
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    # 若 +DM 不大於 -DM 則 +DM=0，反之 -DM=0
    plus_dm = plus_dm.where(high - prev_high > prev_low - low, 0.0)
    minus_dm = minus_dm.where(prev_low - low > high - prev_high, 0.0)
    smooth_tr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=1.0 / period, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=1.0 / period, adjust=False).mean()
    smooth_tr = smooth_tr.replace(0, np.nan)
    plus_di = 100.0 * smooth_plus / smooth_tr
    minus_di = 100.0 * smooth_minus / smooth_tr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx.fillna(0.0)


def _load_daily_close(path, date_col='date', close_col='close'):
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if date_col not in df.columns and 'Date' in df.columns:
        date_col = 'Date'
    if close_col not in df.columns and 'Close' in df.columns:
        close_col = 'Close'
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    return df[[date_col, close_col]].dropna().sort_values(date_col).reset_index(drop=True)


def _align_by_index(
    mtx,
    daily_df,
    date_col,
    day_offset,
    night_offset,
    missing_same_day_offset_adjust: int = 0,
):
    """
    依「交易日序列的 row index」對齊：先找到 date 對應的 row index，再取 index+day_offset（日盤）或 index+night_offset（夜盤）。
    day_offset / night_offset 通常為 -1 或 -2，表示前 1 或前 2 個交易日。
    daily_df 需有 date 與 close，且已按 date 排序、index 為 0,1,2,...
    """
    daily = daily_df.sort_values(date_col).reset_index(drop=True)
    daily = daily.rename(columns={date_col: '_ddate', 'close': '_dclose'})
    daily['_idx'] = np.arange(len(daily))
    mtx = mtx.copy()
    mtx['_ord'] = np.arange(len(mtx))
    # fut_to_one_row_per_session 已保證同一個 date：先夜後日；
    # 此處只需穩定依 date 排序，避免再引入額外判斷/排序規則。
    mtx_sorted = mtx.sort_values('date', kind='mergesort')
    merged = pd.merge_asof(
        mtx_sorted[['date', 'type']].rename(columns={'date': '_k'}),
        daily[['_ddate', '_idx']].rename(columns={'_ddate': '_rdate'}),
        left_on='_k',
        right_on='_rdate',
        direction='backward'
    )
    aligned_idx = merged['_idx'].to_numpy()
    # 同日是否精確匹配：找不到「同日」就額外調整 offset（與 RealTimeWork/selector 同規格）
    left_rdate = merged['_rdate'].to_numpy()
    kdate = merged['_k'].to_numpy()
    has_exact = pd.notna(left_rdate) & (left_rdate == kdate)
    offset_adj = np.where(has_exact, 0, missing_same_day_offset_adjust)

    # 判斷夜盤：type=0 => 夜盤
    is_night = (mtx_sorted['type'].values == 0)
    target_idx = np.where(
        is_night,
        aligned_idx + night_offset + offset_adj,
        aligned_idx + day_offset + offset_adj,
    )
    vals = np.full(len(target_idx), np.nan, dtype=float)
    valid = (target_idx >= 0) & (target_idx < len(daily))
    vals[valid] = daily['_dclose'].iloc[target_idx[valid]].values
    out = mtx_sorted.assign(_val=vals).sort_values('_ord')['_val']
    out.index = mtx.index
    return out


def run_preprocess(
    raw_dir='MTX_rawdata',
    raw_glob='*_fut.csv',
    sp500_path='MTX_rawdata/SP500_daily.csv',
    n225_path='MTX_rawdata/N225_daily.csv',
    twse_path='MTX_rawdata/TWSE_daily.csv',
    out_path='dataset/MTX_separate_sessions.csv',
    start_year=2018,
    end_year=2025,
):
    """
    主流程：讀取 MTX *_fut.csv → 一行一盤 → log_return (close_t/close_t-1)、rolling_std_5 → 對齊 n225/sp500/twse → 輸出。
    """
    raw_dir = os.path.join(_SCRIPT_DIR, raw_dir) if not os.path.isabs(raw_dir) else raw_dir
    pattern = os.path.join(raw_dir, raw_glob)
    fut_files = sorted(glob.glob(pattern))
    if not fut_files:
        raise FileNotFoundError(f'找不到 raw 檔: {pattern}')

    # 1) 一行一盤 MTX（先夜後日）
    list_df = []
    for f in fut_files:
        try:
            df = fut_to_one_row_per_session(f)
            df['date'] = pd.to_datetime(df['date'])
            y = df['date'].dt.year
            df = df[(y >= start_year) & (y <= end_year)]
            if not df.empty:
                list_df.append(df)
        except Exception as e:
            print(f'跳過 {f}: {e}')
    if not list_df:
        raise ValueError('沒有讀到任何 MTX 資料')
    mtx = pd.concat(list_df, ignore_index=True)
    # session 的夜/日順序已在 fut_to_one_row_per_session 階段保證；
    # 此處只做穩定依 date 排序即可（避免再打亂同一天內順序）。
    mtx = mtx.sort_values('date', kind='mergesort').reset_index(drop=True)

    # 補齊 2026：若 futures raw *_fut.csv 只到 2025，則從 mtx_daily_sep.csv 追加 2026 的 night/day。
    # 這一步必須在 log_return/rolling/技術指標計算之前，否則序列語意會錯。
    if end_year >= 2026:
        mtx_daily_sep_path = os.path.join(raw_dir, 'mtx_daily_sep.csv')
        if os.path.isfile(mtx_daily_sep_path):
            daily_sep = pd.read_csv(mtx_daily_sep_path)
            daily_sep['date'] = pd.to_datetime(daily_sep['date'], errors='coerce').dt.normalize()
            if 'session' in daily_sep.columns and 'contract_month' in daily_sep.columns:
                daily_sep['type'] = np.where(
                    daily_sep['session'].astype(str).str.lower().eq('night'),
                    0,
                    1,
                ).astype(int)
                daily_sep['stock'] = 'MTX'
                daily_sep = daily_sep.rename(columns={'contract_month': 'contract month'})
                daily_sep = daily_sep[
                    ['date', 'stock', 'contract month', 'open', 'high', 'low', 'close', 'volume', 'type']
                ].copy()
            else:
                raise ValueError(f'{mtx_daily_sep_path} 欄位格式不符：需含 session, contract_month')

            # 只補 2026 且落在你要的 year 範圍內
            daily_sep = daily_sep[(daily_sep['date'].dt.year == 2026)]

            # 避免同 (date,type) 重複：保留最後一筆
            daily_sep = (
                daily_sep.sort_values(['date', 'type'])
                .drop_duplicates(subset=['date', 'type'], keep='last')
                .reset_index(drop=True)
            )

            existing_keys = mtx[['date', 'type']].drop_duplicates()
            daily_sep = daily_sep.merge(
                existing_keys.assign(_exists=1),
                on=['date', 'type'],
                how='left',
            )
            daily_sep = daily_sep[daily_sep['_exists'].isna()].drop(columns=['_exists'])

            if len(daily_sep) > 0:
                mtx = pd.concat([mtx, daily_sep], ignore_index=True)
                # 同一天先夜(type=0)後日(type=1)
                mtx = mtx.sort_values(['date', 'type'], kind='mergesort').reset_index(drop=True)

    # 2) log_return = close_t / close_t-1（跨盤 close-to-close）
    mtx['close_prev'] = mtx['close'].shift(1)
    mtx['log_return'] = np.log(mtx['close'] / mtx['close_prev']).fillna(0.0)
    mtx['rolling_std'] = mtx['log_return'].rolling(window=5).std().fillna(0.0)
    mtx = mtx.drop(columns=['close_prev'])

    # 2b) 技術指標：MACD, MFI, CCI, ADX（以 session 序列計算，幫助預測 log_return）
    mtx['macd'] = _add_macd(mtx)
    mtx['mfi'] = _add_mfi(mtx)
    mtx['cci'] = _add_cci(mtx)
    mtx['adx'] = _add_adx(mtx)
    # 將 NaN 填 0（前幾筆 rolling 不足）
    for col in ['macd', 'mfi', 'cci', 'adx']:
        mtx[col] = mtx[col].fillna(0.0)

    # 3) 對齊 n225, sp500, twse：依「交易日序列的 row index」取前 1 或前 2 筆（不用 Timedelta，避開休盤日）
    # 先找到每個 MTX.date 在該指數中對應的 row index，再取 index-1 或 index-2
    # SP500: 日盤 -> 前 1 個交易日, 夜盤 -> 前 2 個
    # N225:  日盤/夜盤 -> 前 1 個交易日
    # TWSE:  日盤 -> 當天(0), 夜盤 -> 前 1 個
    mtx['date'] = pd.to_datetime(mtx['date']).dt.normalize()

    def _resolve_path(p):
        return os.path.join(_SCRIPT_DIR, p) if not os.path.isabs(p) else p

    for name, path, day_offset, night_offset, missing_same_day_offset_adjust in [
        ('sp500', sp500_path, -1, -2, 1),
        ('n225', n225_path, -1, -1, 1),
        ('twse', twse_path, 0, -1, 1),
    ]:
        full_path = _resolve_path(path)
        if not os.path.isfile(full_path):
            print(f'警告: 找不到 {name} 檔 {full_path}，該欄填 NaN')
            mtx[name] = np.nan
            continue
        daily = _load_daily_close(full_path)
        if daily.columns[1] != 'close':
            daily = daily.rename(columns={daily.columns[1]: 'close'})
        date_col = daily.columns[0]
        mtx[name] = _align_by_index(
            mtx,
            daily,
            date_col,
            day_offset,
            night_offset,
            missing_same_day_offset_adjust=missing_same_day_offset_adjust,
        )

    # 盤別特徵：此 preprocess 僅輸出 type（夜盤/日盤），下游以 type 做 session embedding 即可

    # 4) 輸出欄位順序
    out_cols = [
        'date', 'stock', 'contract month', 'open', 'high', 'low', 'close', 'volume',
        'log_return', 'rolling_std',
        'macd', 'mfi', 'cci', 'adx',
        'n225', 'sp500', 'twse', 'type'
    ]
    mtx = mtx[[c for c in out_cols if c in mtx.columns]]
    out_path = _resolve_path(out_path)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    mtx.to_csv(out_path, index=False)
    print(f'已寫入: {out_path} ({len(mtx)} 行)')
    return mtx


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MTX 一行一盤 preprocess（separate_model_spec）')
    parser.add_argument('--raw_dir', default='../MTX_rawdata', help='raw 資料目錄（含 *_fut.csv 與 SP500/N225/TWSE 日線）')
    parser.add_argument('--raw_glob', default='*_fut.csv', help='MTX 檔名 glob')
    parser.add_argument('--sp500', default='../MTX_rawdata/SP500_daily.csv', help='SP500 日線 (date, close)')
    parser.add_argument('--n225', default='../MTX_rawdata/N225_daily.csv', help='N225 日線 (date, close)')
    parser.add_argument('--twse', default='../MTX_rawdata/TWSE_daily.csv', help='TWSE 日線 (date, close)')
    parser.add_argument('--output', '-o', default='dataset/MTX_separate_sessions.csv', help='輸出 CSV')
    parser.add_argument('--start_year', type=int, default=2018)
    parser.add_argument('--end_year', type=int, default=2026)
    args = parser.parse_args()
    run_preprocess(
        raw_dir=args.raw_dir,
        raw_glob=args.raw_glob,
        sp500_path=args.sp500,
        n225_path=args.n225,
        twse_path=args.twse,
        out_path=args.output,
        start_year=args.start_year,
        end_year=args.end_year,
    )
