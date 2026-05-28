# -*- coding: utf-8 -*-
"""
即時 15m K → 環境互動 → 輸出 action 的整合腳本
- Shioaji 訂閱 tick，聚合 15 分 K（完成才輸出）
- 每根 K 完成就呼叫 RL 模型 predict，並回寫 action
- 支援 MaskablePPO 的 action mask（若無 mask 也能跑）
"""
import os
import sys
import time
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
from argparse import ArgumentParser
import MTXEnv
import numpy as np
import pandas as pd
import shioaji as sj
from shioaji import TickFOPv1, Exchange
# ===== 你需要的 RL 套件（有 mask 就用 sb3_contrib，沒有就用 sb3）=====
from sb3_contrib import MaskablePPO
import gymnasium as gym
from datetime import datetime
import torch
import random
import yaml
from shioaji.constant import Action, FuturesPriceType, OrderType, FuturesOCType
import tkinter as tk

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)   
    
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

config = load_config("api_key.yaml")
# ===== 你的帳密 / CA（如無 CA 可不啟用）=====
PERSON_ID = config['Future']['api_key']
PASSWORD = config['Future']['secret_key']
CA_PATH = f"./cert/{config['Future']['cert']}.pfx"  # 憑證路徑
CA_PASSWD = config['Future']['passwd']  # 通常是身分證號大寫

# ===== 商品設定：例：股票2330 或期貨 MTX 當月 =====
SYMBOL_TYPE = "FUT"  # "STK" or "FUT"
SYMBOL      = "MXF"  # "2330" for stock; "MTX" for futures root
FUT_MONTH   = "202601"  # 期貨明確月份（建議填明確月份）
BAR_FREQ = "15min"
TZ = "Asia/Taipei"

# 與 test_shioaji_sim_place_order 相同：callback 等待逾時（秒），避免永遠卡住
ORDER_CB_MAX_WAIT_SEC = 120.0

# ===== 模型與 VecNormalize 路徑 =====
MODEL_PATH = r"D:\Quantitative Trading\models\ppo_latest.zip"  # 你的已訓練模型
VECNORM_PATH = r""  # 若你有 VecNormalize，填路徑；沒有就留空

# ===== 輸出動作紀錄 =====
LOG_CSV = "trade/action.csv"

# ===== 你環境的設定（🔧 TODO 看註解）=====
WINDOW_BARS = 64  # 你模型觀測需要幾根歷史K，若環境自己管就忽略
USE_ENV_DIRECT = True  # True: 用你原本的 env 直接 step；False: 用外部組 obs（示範）
MAX_HOLD_SHARES = 2  # 最多持有口數（|shares| <= 2）


# ===================== live window for showing kbars and action ===============

class LiveWindow:
    def __init__(self, poll_ms=200):
        self.root = tk.Tk()
        self.root.title("Live Trading Output")
        self.text = tk.Text(self.root, height=100, width=200)
        self.text.pack()
        self.q = queue.Queue()
        self.poll_ms = poll_ms
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._closing = False
        self.root.after(self.poll_ms, self._drain_queue)

    def _drain_queue(self):
        # 只能在主執行緒呼叫 Tkinter API；這裡安全地把 queue 內容刷到視窗
        try:
            while True:
                msg = self.q.get_nowait()
                self.text.insert(tk.END, msg + "\n")
                self.text.see(tk.END)
        except queue.Empty:
            pass
        if not self._closing:
            self.root.after(self.poll_ms, self._drain_queue)

    def log(self, msg: str):
        # 從任何執行緒都可以呼叫；只放入 queue，不直接碰 UI
        self.q.put(str(msg))

    def on_close(self):
        self._closing = True
        self.root.quit()
        self.root.destroy()

    def run(self):
        # 必須在主執行緒呼叫
        self.root.mainloop()
# ===================== 15 分 K 聚合器 =====================
class BarBuilder15m:
    def __init__(self, freq="15T", tz="Asia/Taipei"):
        self.freq = freq
        self.tz = tz
        self.cur_bin: Optional[pd.Timestamp] = None
        self.buffer: list = []
        self.output_q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.lock = threading.Lock()

    def _finalize_bar(self, bin_label: pd.Timestamp):
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer, columns=["ts", "price", "volume"])
        k = {
            "date": (pd.Timestamp(bin_label) +
                         pd.tseries.frequencies.to_offset(self.freq)),
            "open": float(df["price"].iloc[0]),
            "high": float(df["price"].max()),
            "low":  float(df["price"].min()),
            "close":float(df["price"].iloc[-1]),
            "volume": float(df["volume"].sum()),
        }
        self.output_q.put(k)
        self.buffer.clear()

    def add_tick(self, ts: pd.Timestamp, price: float, volume: float):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=pd.Timestamp.now(tz=self.tz).tzinfo)
        ts = pd.Timestamp(ts).tz_convert(self.tz)
        bin_label = ts.floor(self.freq)
        with self.lock:
            if self.cur_bin is None:
                self.cur_bin = bin_label
            elif bin_label != self.cur_bin:
                self._finalize_bar(self.cur_bin)
                self.cur_bin = bin_label
            self.buffer.append((ts, price, volume))

    def force_flush(self):
        with self.lock:
            if self.cur_bin is not None:
                self._finalize_bar(self.cur_bin)
                self.cur_bin = None

    def poll_finished_bars(self):
        items = []
        while not self.output_q.empty():
            items.append(self.output_q.get())
        return items

# ===================== Shioaji 連線與 Tick 回呼 =====================

bar_builder = BarBuilder15m(freq=BAR_FREQ, tz=TZ)
api = sj.Shioaji()


def _deal_qty(d: Any) -> int:
    q = getattr(d, "quantity", None)
    if q is not None:
        return int(q)
    if isinstance(d, dict):
        return int(d.get("quantity", 0) or 0)
    return 0


def filled_quantity_from_trade(trade: Any) -> int:
    """加總 Shioaji Trade.status.deals 成交量（IOC 可能部分成交）。"""
    st = getattr(trade, "status", None)
    deals = getattr(st, "deals", None) if st is not None else None
    if not deals:
        return 0
    return sum(_deal_qty(d) for d in deals)


_TERMINAL_ORDER_STATUSES = frozenset({"filled", "partfilled", "failed", "cancelled"})

_cb_lock = threading.Lock()
_wait: Optional[Dict[str, Any]] = None  # 下單後等 callback：{ev, oid, want, fill, seen}
# place_order 尚未回傳、_wait 未掛上前先到達的 Order/Deal（Shioaji 會同步觸發 callback）
_cb_buf: "deque[tuple[Any, Any]]" = deque(maxlen=512)
# worker_loop 下單時設為 live_win.log，供 callback 印 price / quantity（與 test 一致）
_RT_ORDER_CB_LOG: Optional[Callable[[str], None]] = None


def _rt_order_cb_trace(msg: str) -> None:
    fn = _RT_ORDER_CB_LOG
    if fn is not None:
        fn(msg)


def _cb_price_qty_line(stat: Any, msg: Any, note: str = "") -> str:
    lab = type(stat).__name__ if stat is not None else "None"
    price = _g(msg, "price")
    if price is None:
        price = _g(msg, "order", "price")
    qty = _g(msg, "quantity")
    if qty is None:
        qty = _g(msg, "order", "quantity")
    tid = _g(msg, "trade_id")
    oidm = _g(msg, "order", "id") or _g(msg, "status", "id")
    suf = f" {note}" if note else ""
    return (
        f"[callback]{suf} stat={lab} price={price!r} quantity={qty!r} "
        f"trade_id={tid!r} order/status_id={oidm!r}"
    )


def _order_status_name(trade: Any) -> str:
    st = getattr(trade, "status", None)
    if st is None:
        return ""
    inner = getattr(st, "status", None)
    if inner is None:
        return ""
    name = getattr(inner, "name", None)
    if name:
        return str(name).strip().lower()
    s = str(inner)
    return s.rsplit(".", 1)[-1].strip().lower() if "." in s else s.strip().lower()


def _g(msg: Any, *keys: str) -> Any:
    x = msg
    for k in keys:
        if x is None:
            return None
        x = x.get(k) if isinstance(x, dict) else getattr(x, k, None)
    return x


def _status_token(x: Any) -> str:
    if x is None:
        return ""
    t = getattr(x, "name", None) or str(x)
    t = str(t).strip().lower()
    return t.rsplit(".", 1)[-1] if "." in t else t


def _dispatch_fop_order(stat: Any, msg: Any, w: Dict[str, Any]) -> None:
    """依 msg 欄位分流（不依賴 type(stat)，避免 FOrder/FDeal 命名差異與 dict 格式）。"""
    if msg is None:
        return
    _rt_order_cb_trace(_cb_price_qty_line(stat, msg))
    tid = _g(msg, "trade_id")
    if tid is not None and str(tid) == w["oid"]:
        dk = str(_g(msg, "exchange_seq") or _g(msg, "seqno") or "")
        if dk in w["seen"]:
            return
        w["seen"].add(dk or "_")
        w["fill"] += int(_g(msg, "quantity") or 0)
        if w["fill"] >= abs(w["want"]):
            w["ev"].set()
        return
    oidm = str(_g(msg, "order", "id") or _g(msg, "status", "id") or "")
    if oidm != w["oid"]:
        return
    sn = _status_token(_g(msg, "status", "status"))
    oq = int(_g(msg, "status", "order_quantity") or _g(msg, "order", "quantity") or 0)
    cq = int(_g(msg, "status", "cancel_quantity") or 0)
    if sn in _TERMINAL_ORDER_STATUSES or (oq and w["fill"] + cq >= oq):
        w["ev"].set()


def _fop_order_cb(stat: Any, msg: Any) -> None:
    """期貨 Order/Deal callback；_wait 未就緒時先寫入 _cb_buf，避免 place_order 回傳前漏單。"""
    if msg is None:
        return
    with _cb_lock:
        w = _wait
        if w is not None:
            _dispatch_fop_order(stat, msg, w)
        else:
            _cb_buf.append((stat, msg))


def wait_order_fill_cb(trade: Any, want: int) -> int:
    """
    set_order_callback 累計成交；與 test_shioaji_sim_place_order 相同模式：
    _cb_buf 接住 place_order 回傳前的 callback、掛 _wait 後重播、逾時跳出、並與 trade.deals 取 max。
    """
    if trade is None:
        return 0
    oid = getattr(getattr(trade, "order", None), "id", None)
    if not oid:
        n = filled_quantity_from_trade(trade)
        _rt_order_cb_trace(f"[callback] 無 order.id，不等待 callback，deals_sum={n}")
        return min(int(n), abs(int(want)))
    ev = threading.Event()
    box: Dict[str, Any] = {"ev": ev, "oid": str(oid), "want": int(want), "fill": 0, "seen": set()}
    global _wait
    with _cb_lock:
        _wait = box
        while _cb_buf:
            st, m = _cb_buf.popleft()
            _dispatch_fop_order(st, m, box)
    try:
        t0 = time.monotonic()
        while not ev.is_set() and not stop_event.is_set():
            if time.monotonic() - t0 > ORDER_CB_MAX_WAIT_SEC:
                _rt_order_cb_trace(
                    f"[callback] 逾時 {ORDER_CB_MAX_WAIT_SEC}s oid={oid!r} "
                    f"fill={box['fill']} status={_order_status_name(trade)!r}"
                )
                break
            ev.wait(0.2)
        sync = filled_quantity_from_trade(trade)
        return min(max(int(box["fill"]), sync), abs(int(want)))
    finally:
        with _cb_lock:
            if _wait is box:
                _wait = None


def place_futures_order(net_contracts: int) -> Any:
    """
    下 MXF 近月 R1 期貨單（與 main 訂閱的合約一致）。
    net_contracts > 0 買進、< 0 賣出；== 0 不送單，回傳 None。
    """
    if net_contracts == 0:
        return None
    q = abs(int(net_contracts))
    contract = api.Contracts.Futures[f"{SYMBOL}R1"]
    order = api.Order(
        action=Action.Buy if net_contracts > 0 else Action.Sell,
        price=0,
        quantity=q,
        price_type=FuturesPriceType.MKT,
        order_type=OrderType.IOC,
        octype=FuturesOCType.Auto,
        account=api.futopt_account,
    )
    return api.place_order(contract, order)


def on_tick_fop(exchange, tick):
    ts = pd.Timestamp(tick.datetime, tz=TZ)
    hhmm = ts.strftime("%H:%M")
    if (hhmm >= "05:00" and hhmm < "08:45") or (hhmm >= "13:45" and hhmm < "15:00"):
        return  # 不加進 bar_builder
    bar_builder.add_tick(tick.datetime, float(tick.close), float(tick.volume))

def make_env(env_name, save_dir, sim_start, sim_end, train_start, train_end, init_balance=1e6):
    # environment
    env_id = env_name
    env = gym.make(env_id, save_dir=save_dir, sim_start=sim_start, sim_end=sim_end, train_start=train_start, train_end=train_end, init_balance=init_balance, margins=[131500,100750,97250])
        
    return env
# ===================== 環境橋接（🔧 只改這些） =====================
@dataclass
class LiveBar:
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

# ===================== 模型載入與推論 =====================
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MODEL_PATH 不存在：{model_path}")
    model = MaskablePPO.load(model_path)  # 如要用 GPU 改 device="cuda"
    return model


def clip_action_to_max_hold(raw_action: Any, current_shares: int, max_hold: int = MAX_HOLD_SHARES) -> int:
    """
    將模型 action（0..200，100=hold）裁剪為不超過最大持倉。
    env.step 內部會用 delta = action - 100，因此限制條件為：
    |current_shares + delta| <= max_hold
    """
    a = int(np.asarray(raw_action).item())
    delta = a - 100
    lo = -max_hold - int(current_shares)
    hi = max_hold - int(current_shares)
    delta = int(np.clip(delta, lo, hi))
    return 100 + delta

stop_event = threading.Event()

# ===================== 主工作執行緒：處理完成的 K → 模型互動 =====================
def worker_loop(trade_env, model, agent, final_day, live_win, live_order: bool = False):
    global _RT_ORDER_CB_LOG

    print("✅ 交易迴圈已啟動：每根 15m K 完成就推論動作")
    live_win.log("Start trading")
    while not stop_event.is_set():
        done_bars = bar_builder.poll_finished_bars()
        
        if not done_bars:
            if datetime.now().strftime("%H:%M") == "13:45" or datetime.now().strftime("%H:%M") == "05:00":
                bar_builder.force_flush()
            time.sleep(0.25)
            continue
        live_win.log(done_bars)
        print(done_bars)
        for b in done_bars:
            obs = trade_env.update_state(b)
            if final_day and b['date'].strftime("%H:%M") == "13:30":
                trade_env.finalize_episode()
                stop_event.set()
                break
            if agent == "long":
                raw_action, _ = model.predict(obs, deterministic=False, action_masks=trade_env.valid_action_mask_long())
            elif agent == "short":
                raw_action, _ = model.predict(obs, deterministic=False, action_masks=trade_env.valid_action_mask_short())
            else:
                raw_action = 100 - int(trade_env.shares)
            #if (b['date'].strftime("%H:%M") == "04:45" or b['date'].strftime("%H:%M") == "13:30"):
            #    raw_action = 100 - int(trade_env.shares)
            raw_action = int(np.asarray(raw_action).item())
            # 僅在送進環境前限制最大持倉，不更動 model.predict 輸入/抽樣流程
            raw_action = clip_action_to_max_hold(raw_action, int(trade_env.shares), MAX_HOLD_SHARES)
            desired_net = raw_action - 100  # 與 env 一致：index 中心 100

            if live_order and desired_net != 0:
                try:
                    _RT_ORDER_CB_LOG = live_win.log
                    trade = place_futures_order(desired_net)
                    filled = wait_order_fill_cb(trade, desired_net)
                    if filled <= 0:
                        live_win.log(
                            f"[live] 成交 0 desired_net={desired_net} "
                            f"status={getattr(getattr(trade, 'status', None), 'status', None)}"
                        )
                        raw_action = 100
                    else:
                        cap = min(filled, abs(desired_net))
                        actual_net = cap if desired_net > 0 else -cap
                        raw_action = 100 + actual_net
                        live_win.log(f"[live] desired_net={desired_net} filled={filled} step_net={actual_net}")
                except Exception as e:
                    live_win.log(f"[live] place_order 失敗: {e!r}")
                    raw_action = 100
                finally:
                    _RT_ORDER_CB_LOG = None

            obs, reward, done, _, info = trade_env.step(raw_action)
            live_win.log(info)
            print(info)
            if(info['date'].split()[1] == "13:45:00" or info['date'].split()[1] == "05:00:00"):
                trade_env.save_results()
                stop_event.set()
    live_win.log("End trading")            
    print("收盤")
    live_win.on_close()  
        

# ===================== 程式進入點 =====================
def main():
    # 1) 登入 Shioaji
    parser = ArgumentParser()
    parser.add_argument('--agent', type=str, help="agent type")
    parser.add_argument('--final_day', action="store_true", help="final day of R1 contract")
    parser.add_argument('--config', type=str, help="model config file")
    parser.add_argument(
        "--live-order",
        action="store_true",
        help="實際呼叫 Shioaji 下單；未指定則僅用模型 action 做 env step（不下單）",
    )
    parser.add_argument('--contract', required=True, type=str, help="contract month")
    args = parser.parse_args()
    seed_everything(42)
    api.login(PERSON_ID, PASSWORD, contracts_cb=lambda st: print(f"{st} ready!"))
    api.activate_ca(ca_path=CA_PATH, ca_passwd=CA_PASSWD)
    api.set_order_callback(lambda *a: _fop_order_cb(a[0], a[1]) if len(a) >= 2 else None)
    model_config = load_config(args.config)
    contract = api.Contracts.Futures[f"{SYMBOL}R1"]
    api.quote.set_on_tick_fop_v1_callback(on_tick_fop)

    api.quote.subscribe(contract, quote_type="tick", version="v1")
    print(f"🚀 已訂閱 {SYMBOL} ticks，開始聚合 15m K")

    # 3) 載入模型與橋接
    trade_env = make_env("MTXEnvRealTimeTrader", "RealTimeWork/trade", args.contract, args.contract, model_config['date']['train_start'], model_config['date']['train_end'], 1e6)
    if args.agent == "long":
        model = load_model(f"trained_model/long_{model_config['date']['test_start']}_step2_MFI8.zip")
        trade_env.reset(options=1)
    elif args.agent == "short":
        model = load_model(f"trained_model/short_{model_config['date']['test_start']}_step2_MFI8.zip")
        trade_env.reset(options=-1)
    else:
        model = None
        trade_env.reset(options=0)
    
    live_win = LiveWindow(poll_ms=200)
    # 4) 啟動工作執行緒
    t = threading.Thread(
        target=worker_loop,
        args=(trade_env, model, args.agent, args.final_day, live_win, bool(args.live_order)),
        daemon=True,
    )
    t.start()
    live_win.run()
    # 5) 主執行緒維持；必要時可在收盤強制收斂
    api.quote.unsubscribe(contract, quote_type="tick", version="v1")
    api.logout()

if __name__ == "__main__":
    main()
