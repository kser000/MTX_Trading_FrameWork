import gymnasium as gym
import MTXEnv
import numpy as np
import torch
import random
from argparse import ArgumentParser
from sb3_contrib import MaskablePPO
import pandas as pd
import yaml
from datetime import datetime, timedelta
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
import sys
import os

_FW_ROOT = Path(__file__).resolve().parent / "期貨常見策略"
if str(_FW_ROOT) not in sys.path:
    sys.path.insert(0, str(_FW_ROOT))
from src.mtx_system.regime_gate import (  # noqa: E402
    load_daily_regime_by_date,
    normalize_regime,
    summarize_regimes,
)

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
                
def make_env(env_name, **kwargs):
    """
    創建環境，使用 kwargs 傳遞參數
    
    Args:
        env_name: 環境名稱
        **kwargs: 環境參數，包括：
            - save_dir: 保存目錄
            - sim_start: 模擬開始日期
            - sim_end: 模擬結束日期
            - train_start: 訓練開始日期
            - train_end: 訓練結束日期
            - init_balance: 初始資金
            - feature_columns: 特徵欄位
            - data_dir: 數據目錄
            - mode: 模式 (預設 "Test")
    """
    # 設定預設值
    default_params = {
        "mode": "Test",
    }
    
    # 合併預設參數和傳入的參數
    env_params = {**default_params, **kwargs}
    
    # 創建環境
    env = gym.make(env_name, **env_params)
        
    return env


def clip_rl_action_to_max_shares(raw_action, env, max_shares: int):
    """
    將 RL 輸出的 gym action（0..200，100=hold）裁剪後再送進 env.step。
    滿足 |shares + delta| <= max_shares；不改動 predict() 的呼叫方式與觀測。

    注意：強制平倉等由 CombineTrader 直接指定 action 的路徑請勿經過此函式。
    """
    if max_shares <= 0:
        return raw_action
    s = int(env.shares)
    raw = int(np.asarray(raw_action).item())
    d = raw - 100
    lo = -max_shares - s
    hi = max_shares - s
    d_clip = int(np.clip(d, lo, hi))
    return d_clip + 100


def load_selector_test_result(csv_path):
    """
    載入 selector 的 test_result.csv，建立 (date_str, session) -> direction_pred 與 log_return_pred 查表，
    並回傳排序後的交易日列表。若有 direction_true 則一併建立 lookup_true（完美 selector 用）。
    CSV 每日期兩行：先夜盤後日盤（列順序）；欄位需有 date, direction_pred，若有 log_return_pred 則一併載入供波動過濾。
    """
    df = pd.read_csv(csv_path)
    if 'direction_pred' not in df.columns and 'direction' in df.columns:
        df['direction_pred'] = df['direction']
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    lookup = {}
    lookup_log_return = {}
    lookup_true = {}  # (date_str, session) -> direction_true，用於 --selector_use_true（100% 正確率 ROI）
    has_true = 'direction_true' in df.columns
    for d, g in df.groupby('date', sort=False):
        rows = g.reset_index(drop=True)
        if len(rows) >= 1:
            lookup[(d, 'night')] = int(rows.iloc[0]['direction_pred'])
            if 'log_return_pred' in rows.columns:
                lookup_log_return[(d, 'night')] = float(rows.iloc[0]['log_return_pred'])
            if has_true:
                lookup_true[(d, 'night')] = int(rows.iloc[0]['direction_true'])
        if len(rows) >= 2:
            lookup[(d, 'day')] = int(rows.iloc[1]['direction_pred'])
            if 'log_return_pred' in rows.columns:
                lookup_log_return[(d, 'day')] = float(rows.iloc[1]['log_return_pred'])
            if has_true:
                lookup_true[(d, 'day')] = int(rows.iloc[1]['direction_true'])
    trading_dates = sorted(set(k[0] for k in lookup.keys()))
    return lookup, trading_dates, lookup_log_return, (lookup_true if has_true else None)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='The random seed of the system.')
    parser.add_argument('--env', default='MTXEnvTrader', type=str, help='The name of the env.')
    parser.add_argument('--balance', default=1000000, type=int, help='initial balance of agent.')
    parser.add_argument('--a_config', type=str, required=True, help='The config file path for the agent (e.g., config/202501.yaml)')
    parser.add_argument('--test_result_path', type=str, required=True, help='Path to selector test_result.csv (date, direction_pred per session).')
    parser.add_argument('--selector_use_true', action='store_true', help='Use direction_true as selector (perfect 100%% accuracy) to see ROI upper bound; requires test_result_path with direction_true column.')
    parser.add_argument(
        '--max-shares',
        type=int,
        default=None,
        metavar='N',
        help='僅裁剪「RL predict 之後」送進 env 的張數上限（|shares|<=N）；不改 obs/mask，不影響模型輸出計算。平倉/換日強制 flatten 不套用。預設關閉（沿用環境內建上限）。',
    )
    parser.add_argument('--save_dir', type=str, default='CombineTrader_test', help='Directory to save results')
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Data directory')
    parser.add_argument(
        '--regime_daily_csv',
        type=str,
        default=str(_FW_ROOT / "results_regime_simple" / "daily_regime.csv"),
        help='daily_regime.csv (三類：uptrend/downtrend/range)。Selector 與 regime 同向才下單。',
    )
    parser.add_argument(
        '--regime_col',
        type=str,
        default='regime_close',
        help='日線 CSV 的 regime 欄位名（預設 regime_close）。',
    )
    args = parser.parse_args()
    seed_everything(args.seed)
    
    # 加载 agent config
    agent_config = load_config(args.a_config)
    agent_feat_columns = agent_config.get('feat', None)
    
    # set env and trader
    env = make_env(
        args.env,
        save_dir=f'valid_csv/{args.save_dir}',
        sim_start="202601",
        sim_end="202605", 
        train_start=agent_config['date']['train_start'],
        train_end=agent_config['date']['train_end'],
        init_balance=args.balance,
        feature_columns=agent_feat_columns,
        data_dir=args.data_dir
    )
    long_trader = MaskablePPO.load(f'trained_model/long_202507_step2_MFI8.zip')
    short_trader = MaskablePPO.load(f'trained_model/short_202507_step2_MFI8.zip')
    sim_months = 5
    # initial metric list
    rewards = [0] * sim_months
    RoR = [0] * sim_months
    Sharp = [0] * sim_months
    MDD = [0] * sim_months
    
    '''
    '''
    cost = args.balance
    profit = 0
    asset = args.balance
    '''
    '''
    
    total_choice_wrong = 0
    total_trend_num = 0

    # Selector 訊號來源：test_result CSV（必填）
    if not os.path.isfile(args.test_result_path):
        raise FileNotFoundError(f"test_result not found: {args.test_result_path}")
    selector_lookup, selector_trading_dates, selector_log_return, selector_lookup_true = load_selector_test_result(args.test_result_path)
    if args.selector_use_true and selector_lookup_true is not None:
        selector_lookup = selector_lookup_true
        print(f"[Selector] Using direction_true (100%% accuracy mode) -> ROI upper bound")
    elif args.selector_use_true and selector_lookup_true is None:
        raise ValueError("--selector_use_true requires test_result CSV to have direction_true column.")
    VOL_THRESHOLD = 0  # |log_return_pred| < 此值視為波動小，trader=0（不交易）
    print(f"[Selector] Loaded test_result: {args.test_result_path} ({len(selector_lookup)} entries, {len(selector_trading_dates)} trading days), vol_threshold={VOL_THRESHOLD}")

    if args.max_shares is not None:
        print(f"[Exec] max_shares={args.max_shares}（僅限 RL 下單；predict 輸入不變，送 step 前裁剪）")

    # ---- Regime gate（日線三類） ----
    regime_by_date, _ = load_daily_regime_by_date(args.regime_daily_csv, regime_col=args.regime_col)
    print(f"[Regime] {args.regime_daily_csv} ({len(regime_by_date)} days, col='{args.regime_col}')")
    print("[Combine] 兩者皆多→long；皆空→short；其餘→flat")
    regime_flag = None
    regime_stats = {
        'updates_15_15': 0,
        'uptrend': 0,
        'downtrend': 0,
        'range': 0,
        'missing': 0,
        'long': 0,
        'short': 0,
        'flat_disagree': 0,
        'flat_regime_not_trend': 0,
    }

    def apply_regime_gate(curr_trader, time_label, current_date_str):
        """Selector 給的 1/-1/0 與當前 regime_flag 結合。"""
        if curr_trader == 0:
            return 0
        r = normalize_regime(regime_flag)
        if curr_trader == 1:
            if r == 'uptrend':
                regime_stats['long'] += 1
                return 1
            if r == 'downtrend':
                regime_stats['flat_disagree'] += 1
                tag = 'flat_disagree'
            else:
                regime_stats['flat_regime_not_trend'] += 1
                tag = 'flat_regime_not_trend'
            print(f"[{current_date_str} {time_label}] regime={r} blocks LONG ({tag}) -> trader=0")
            env.set_trader(0)
            return 0
        if curr_trader == -1:
            if r == 'downtrend':
                regime_stats['short'] += 1
                return -1
            if r == 'uptrend':
                regime_stats['flat_disagree'] += 1
                tag = 'flat_disagree'
            else:
                regime_stats['flat_regime_not_trend'] += 1
                tag = 'flat_regime_not_trend'
            print(f"[{current_date_str} {time_label}] regime={r} blocks SHORT ({tag}) -> trader=0")
            env.set_trader(0)
            return 0
        return curr_trader

    def set_trader_from_selector(selector_key, time_label):
        """Baseline: use selector to set env.trader (no suspend)."""
        if selector_key is None:
            env.set_trader(0)
            return 0
        trend_pred = selector_lookup.get(selector_key)
        if trend_pred is None:
            trend_pred = 0
        if selector_log_return and selector_key in selector_log_return:
            abs_pred = abs(selector_log_return[selector_key])
            if abs_pred < VOL_THRESHOLD:
                env.set_trader(0)
                print(f"[{selector_key[0]} {time_label}] Selector: |log_return_pred|={abs_pred:.6f}<{VOL_THRESHOLD} -> trader=0 (no trade)")
                return 0
        if trend_pred == 1:
            env.set_trader(1)
            print(f"[{selector_key[0]} {time_label}] Selector: UP -> long_trader")
            return 1
        env.set_trader(-1)
        print(f"[{selector_key[0]} {time_label}] Selector: DOWN -> short_trader")
        return -1

    for j in range(sim_months):
        start_asset = args.balance
        obs, info = env.reset()
        done = False
        time_step = env.index
        curr_trader = 0

        while not done:
            action = 100
            date_time = datetime.strptime(env.df.iloc[time_step]["date"], "%Y-%m-%d %H:%M:%S").strftime("%H:%M")
            current_date_str = datetime.strptime(env.df.iloc[time_step]["date"], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
            if date_time == '15:15' or date_time == '09:00':
                seed_everything(args.seed)

            # 15:15：夜盤開盤
            if date_time == '15:15':
                # 先更新今日 TWII 收盤後的 regime（之後沿用到下一個 15:15）
                regime_flag = normalize_regime(regime_by_date.get(current_date_str))
                regime_stats['updates_15_15'] += 1
                if regime_flag in ('uptrend', 'downtrend', 'range'):
                    regime_stats[regime_flag] += 1
                else:
                    regime_stats['missing'] += 1
                idx = bisect_right(selector_trading_dates, current_date_str)
                next_date_str = selector_trading_dates[idx] if idx < len(selector_trading_dates) else None
                key = (next_date_str, 'night') if next_date_str else None
                curr_trader = set_trader_from_selector(key, '15:15')
                curr_trader = apply_regime_gate(curr_trader, '15:15', current_date_str)
                total_trend_num += 1

            # 09:00：日盤開盤
            if date_time == '09:00':
                key = (current_date_str, 'day')
                curr_trader = set_trader_from_selector(key, '09:00')
                curr_trader = apply_regime_gate(curr_trader, '09:00', current_date_str)
                total_trend_num += 1
            # Choose which trader to use. trader=0：有倉平倉、否則 hold（對應 valid_action_mask_trader0）
            if curr_trader == 1:
                action, state = long_trader.predict(obs, deterministic=False, action_masks=env.valid_action_mask_long())
                if args.max_shares is not None:
                    action = clip_rl_action_to_max_shares(action, env, args.max_shares)
            elif curr_trader == -1:
                action, state = short_trader.predict(obs, deterministic=False, action_masks=env.valid_action_mask_short())
                if args.max_shares is not None:
                    action = clip_rl_action_to_max_shares(action, env, args.max_shares)
            elif curr_trader == 0:
                # 波動小：有 shares 就平倉，否則 hold
                if env.shares == 0:
                    action = 100
                else:
                    action = 100 - env.shares
            
            #if (date_time == '04:45' or date_time == '13:30'):
            #    action = 100 - env.shares
            # Execute action
            obs, reward, done, _, info = env.step(action)
            rewards[j] += reward
            time_step += 1
            
        RoR[j], Sharp[j], MDD[j] = env.get_performance()
        print(f'Sharp: {Sharp[j]}')
        print(f'MDD: {MDD[j]}')
        print(f'ROR : {RoR[j]}')
        
        '''
        '''
        profit = info['balance'] - args.balance         
        print(f'asset : {asset}, profit : {profit}')
        print(f'date : {info["date"]}')

        asset += profit
        if asset < args.balance:
            cost += args.balance - asset
            asset = args.balance
        '''
        '''    
        print(f'j = {j}')

    rewards = np.array(rewards)
    RoR = np.array(RoR)
    Sharp = np.array(Sharp)
    MDD = np.array(MDD)


    ROI = (asset-cost)/cost
    print(f'mean reward : {np.mean(rewards):4.4f}, unbias std_reward : {np.std(rewards, ddof=1):4.4f}, bias std_reward : {np.std(rewards, ddof=0):4.4f}')
    print(f'mean RoR : {np.mean(RoR):4.4f}, std RoR : {np.std(RoR):4.4f}, MDD : {np.min(MDD):4.4f}')
    print(f'total trend decisions : {total_trend_num}')
    print(
        f"[Regime] updates@15:15={regime_stats['updates_15_15']}, "
        f"uptrend={regime_stats['uptrend']}, downtrend={regime_stats['downtrend']}, "
        f"range={regime_stats['range']}, missing={regime_stats['missing']}"
    )
    print(
        f"[Combine] long={regime_stats['long']}, short={regime_stats['short']}, "
        f"flat_disagree={regime_stats['flat_disagree']}, "
        f"flat_regime_not_trend={regime_stats['flat_regime_not_trend']}"
    )
    print(f'ROI : {ROI}')
    mu = RoR.mean()
    sigma = RoR.std(ddof=1)
    print("Sharp ratio:", mu/sigma * np.sqrt(12))
    #print(f'IRR : {IRR}')
    #print(f'Ann Vol : {AVOL}')
    #print(f'Sharp ratio : {(IRR - 0.017) / AVOL}')
    print(f'cost : {cost}')
    print(f'asset : {asset}')
