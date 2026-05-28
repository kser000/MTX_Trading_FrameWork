# -*- coding: utf-8 -*-
"""
Selector class:
- create_dataset(seq_len, pred_date, session): 依 mtx_daily_sep + N225/TWSE/SP500 產生資料，
  回傳「pred 那筆之前」的 seq_len 列（不包含 pred_date/session 那筆），不存檔。
  predict() 會在寫暫存 CSV 時自動補一列合成 pred row，讓 PatchTST 產生單筆樣本。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
try:
    from .utils import add_adx, add_cci, add_macd, add_mfi
except ImportError:
    from utils import add_adx, add_cci, add_macd, add_mfi


def _load_daily_close(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{path} 需含 date, close 欄位")
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def _align_by_index(
    mtx: pd.DataFrame,
    daily_df: pd.DataFrame,
    day_offset: int,
    night_offset: int,
    missing_same_day_offset_adjust: int = 0,
) -> pd.Series:
    daily = daily_df.sort_values("date").reset_index(drop=True).copy()
    daily["_idx"] = np.arange(len(daily))
    mtx_tmp = mtx.copy()
    mtx_tmp["_ord"] = np.arange(len(mtx_tmp))
    mtx_sorted = mtx_tmp.sort_values("date").copy()
    # 對齊規則：
    # - 一律使用左側（<= mtx_date）的最後一筆當錨點（backward）
    # - 若 daily 沒有「同日」資料（非精確 match），offset 另外做平移調整
    merged = pd.merge_asof(
        mtx_sorted[["date", "type"]].rename(columns={"date": "_k"}),
        daily[["date", "_idx"]].rename(columns={"date": "_rdate"}),
        left_on="_k",
        right_on="_rdate",
        direction="backward",
    )

    aligned_idx = merged["_idx"].to_numpy()
    # 用 array 做逐元素比較，避免 pandas 以 index label 對齊後再比較而報錯
    left_rdate = merged["_rdate"].to_numpy()
    kdate = mtx_sorted["date"].to_numpy()
    has_exact = pd.notna(left_rdate) & (left_rdate == kdate)
    is_night = mtx_sorted["type"].values == 0
    # 若同日缺失（has_exact=False），offset 再額外做調整（例如 SP500/N225: -1）
    offset_adj = np.where(has_exact, 0, missing_same_day_offset_adjust)
    target_idx = np.where(
        is_night,
        aligned_idx + night_offset + offset_adj,
        aligned_idx + day_offset + offset_adj,
    )
    vals = np.full(len(target_idx), np.nan, dtype=float)
    valid = (target_idx >= 0) & (target_idx < len(daily))
    vals[valid] = daily["close"].iloc[target_idx[valid]].values
    out = mtx_sorted.assign(_val=vals).sort_values("_ord")["_val"]
    out.index = mtx.index
    return out


class Selector:
    """處理 selector 相關運作；目前先實作 create_dataset。"""

    def __init__(self, project_root: Optional[str | Path] = None) -> None:
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent.parent
        self.raw_dir = self.project_root / "MTX_rawdata"
        self.dataset_path = self.project_root / "PatchTST" / "dataset" / "MTX_separate_sessions.csv"

    def create_dataset(self, seq_len: int, pred_date: Optional[str] = None, session: int | str = 0) -> pd.DataFrame:
        """
        依四個 CSV 建立資料集（不寫檔）。

        - 若 pred_date is None：維持舊行為，直接回傳最後 seq_len 列（與先前 selector 一致）。
        - 若 pred_date 有給：以 pred_date + session(type) 找到 anchor row，回傳 anchor 之前的 seq_len（不含 anchor）。
        """
        if seq_len <= 0:
            raise ValueError("seq_len 必須 > 0")
        session_id = None
        pred_ts = None
        if pred_date is not None:
            pred_ts = pd.Timestamp(pd.to_datetime(pred_date, errors="coerce"))
            if pd.isna(pred_ts):
                raise ValueError(f"pred_date 無法解析: {pred_date!r}")
            pred_ts = pred_ts.normalize()
            if isinstance(session, str):
                s = session.strip().lower()
                if s in ("night", "n", "0"):
                    session_id = 0
                elif s in ("day", "d", "1"):
                    session_id = 1
                else:
                    raise ValueError(f"session 無法解析（請用 0/1 或 night/day）: {session!r}")
            else:
                session_id = int(session)
            if session_id not in (0, 1):
                raise ValueError(f"session 必須為 0/1，目前為 {session_id}")

        mtx_path = self.raw_dir / "mtx_daily_sep.csv"
        if not mtx_path.is_file():
            raise FileNotFoundError(f"找不到 {mtx_path}")

        mtx = pd.read_csv(mtx_path)
        required = ["date", "session", "open", "high", "low", "close", "volume", "contract_month"]
        miss = sorted(set(required) - set(mtx.columns))
        if miss:
            raise ValueError(f"{mtx_path} 缺少欄位: {miss}")

        mtx = mtx[required].copy()
        mtx["date"] = pd.to_datetime(mtx["date"], errors="coerce").dt.normalize()
        mtx = mtx.dropna(subset=["date"])
        mtx["stock"] = "MTX"
        mtx["contract month"] = mtx["contract_month"].astype(str)
        # 與 MTX_separate_sessions / PatchTST：0=夜盤(night), 1=日盤(day)
        mtx["type"] = np.where(mtx["session"].astype(str).str.lower().eq("night"), 0, 1).astype(int)
        # 同一曆日內與 mtx_daily_sep 一致：先夜盤(0)再日盤(1)；shift(1) 才是「上一時段收盤」。
        # 若只 sort date，同日日/夜順序不定，log_return 會錯。
        mtx = mtx.sort_values(["date", "type"], ascending=[True, True]).reset_index(drop=True)

        # close-to-close 特徵
        mtx["close_prev"] = pd.to_numeric(mtx["close"], errors="coerce").shift(1)
        mtx["log_return"] = np.log(pd.to_numeric(mtx["close"], errors="coerce") / mtx["close_prev"]).fillna(0.0)
        mtx["rolling_std"] = mtx["log_return"].rolling(window=5).std().fillna(0.0)
        mtx = mtx.drop(columns=["close_prev"])

        # 技術指標
        for c in ["open", "high", "low", "close", "volume"]:
            mtx[c] = pd.to_numeric(mtx[c], errors="coerce")
        mtx["macd"] = add_macd(mtx).fillna(0.0)
        mtx["mfi"] = add_mfi(mtx).fillna(0.0)
        mtx["cci"] = add_cci(mtx).fillna(0.0)
        mtx["adx"] = add_adx(mtx).fillna(0.0)

        # 外盤對齊偏移（沿用既有規則）
        sp500 = _load_daily_close(self.raw_dir / "SP500_daily.csv")
        n225 = _load_daily_close(self.raw_dir / "N225_daily.csv")
        twse = _load_daily_close(self.raw_dir / "TWSE_daily.csv")
        # 若指數日線缺少「同日」資料，offset 往右平移一格（+1）
        # 例如 SP500: (night, day)=(-2, -1) -> (-1, 0)
        mtx["sp500"] = _align_by_index(mtx, sp500, day_offset=-1, night_offset=-2, missing_same_day_offset_adjust=1)
        mtx["n225"] = _align_by_index(mtx, n225, day_offset=-1, night_offset=-1, missing_same_day_offset_adjust=1)
        mtx["twse"] = _align_by_index(mtx, twse, day_offset=0, night_offset=-1, missing_same_day_offset_adjust=1)

        out_cols = [
            "date",
            "stock",
            "contract month",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "log_return",
            "rolling_std",
            "macd",
            "mfi",
            "cci",
            "adx",
            "n225",
            "sp500",
            "twse",
            "type",
        ]
        out = mtx[out_cols].copy()
        out["_type_ord"] = out["type"].astype(int)
        out = (
            out.sort_values(["date", "_type_ord"])
            .drop_duplicates(subset=["date", "type"], keep="last")
            .drop(columns=["_type_ord"])
            .reset_index(drop=True)
        )
        # 舊模式：直接拿最後 seq_len 列（不含任何 pred info）
        if pred_date is None:
            legacy = out.tail(seq_len).copy()
            legacy["date"] = pd.to_datetime(legacy["date"]).dt.strftime("%Y-%m-%d")
            return legacy.reset_index(drop=True)

        d = pd.to_datetime(out["date"]).dt.normalize()
        # 找到 pred_date 的 rows 區間，再從中挑出指定 session 的那列作為 anchor
        i0 = int(d.searchsorted(pred_ts, side="left"))
        i1 = int(d.searchsorted(pred_ts, side="right"))
        if i1 <= i0:
            raise ValueError(f"pred_date={pred_ts.date()} 在資料中無任何列（i0={i0}, i1={i1}）")
        day_block = out.iloc[i0:i1].reset_index(drop=True)
        hit = np.where(day_block["type"].astype(int).to_numpy() == session_id)[0]
        if len(hit) == 0:
            raise ValueError(f"pred_date={pred_ts.date()} 找不到 session(type)={session_id} 的資料列")
        # 正常情況每個日期只有 0/1 各一列；若多列則用第一列
        anchor_idx = i0 + int(hit[0])
        border1 = anchor_idx - seq_len
        if border1 < 0:
            raise ValueError(
                f"pred_date={pred_ts.date()} session={session_id} 之前歷史不足：需要至少 {seq_len} 列，目前 border1={border1}"
            )
        window = out.iloc[border1:anchor_idx].copy()
        window["date"] = pd.to_datetime(window["date"]).dt.strftime("%Y-%m-%d")
        # 記錄 anchor 資訊（predict() 會用）
        ret = window.reset_index(drop=True)
        ret.attrs["pred_date"] = pred_ts.strftime("%Y-%m-%d")
        ret.attrs["pred_session_id"] = int(session_id)
        return ret

    def predict(
        self,
        *,
        model_id: str = "MTX_spot_separate_128_embed",
        seq_len: Optional[int] = None,
        pred_date: Optional[str] = None,
        session: int | str = 0,
    ) -> dict:
        """
        使用指定模型做預測。
        流程：
        1) create_dataset(seq_len, pred_date) 產生與 Dataset_Pred 對齊的切片
        2) 寫入暫存 CSV（非正式資料集）
        3) 呼叫 run_longExp.py --do_predict（帶 --pred_start/--pred_end 與訓練一致）
        4) 讀取 prediction_result.csv，回傳最後一筆
        """
        def _infer_seq_len(mid: str) -> int:
            # 例如：MTX_spot_separate_128_embed -> 128
            parts = str(mid).replace("-", "_").split("_")
            for tok in parts:
                if tok.isdigit():
                    return int(tok)
            return 128

        if seq_len is None:
            seq_len = _infer_seq_len(model_id)

        if seq_len <= 0:
            raise ValueError(f"seq_len 必須 > 0，目前為 {seq_len}")

        data_cols = "log_return rolling_std macd mfi cci adx n225 sp500 twse"

        df = self.create_dataset(seq_len=seq_len, pred_date=pred_date, session=session)
        if df.empty:
            raise ValueError("create_dataset 回傳空資料，無法預測")
        if len(df) != seq_len:
            raise ValueError(f"create_dataset 長度需為 seq_len（不含 pred row），目前 len={len(df)}, seq_len={seq_len}")
        # 不給 pred_start/end：Dataset_Pred 已支援 len(data)==seq_len 時回傳 1 筆樣本
        df_for_predict = df

        supervised_dir = self.project_root / "PatchTST" / "PatchTST_supervised"
        run_script = supervised_dir / "run_longExp.py"
        if not run_script.is_file():
            raise FileNotFoundError(f"找不到 {run_script}")

        temp_dir = self.project_root / "PatchTST" / "dataset" / "_selector_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_csv = temp_dir / f"{model_id}_predict_input.csv"
        df_for_predict.to_csv(temp_csv, index=False)

        scaler_path = supervised_dir / "checkpoints" / model_id / "scaler.pkl"
        cmd = [
            sys.executable,
            "run_longExp.py",
            "--is_training",
            "0",
            "--do_predict",
            "--root_path",
            str(temp_dir) + "/",
            "--data_path",
            temp_csv.name,
            "--model_id",
            model_id,
            "--model",
            "PatchTST",
            "--data",
            "custom",
            "--features",
            "MS",
            "--target",
            "log_return",
            "--seq_len",
            str(seq_len),
            "--cols",
            data_cols,
            "--enc_in",
            "9",
            "--scaler_path",
            str(scaler_path),
            "--e_layers",
            "3",
            "--n_heads",
            "4",
            "--d_model",
            "128",
            "--d_ff",
            "256",
            "--patch_len",
            "4",
            "--stride",
            "2",
            "--padding_patch",
            "end",
            "--revin",
            "1",
            "--freq",
            "d",
            "--use_session_embed",
            "1",
        ]
        if not scaler_path.is_file():
            print(f"警告: 找不到 scaler，將使用預測資料自行 fit: {scaler_path}")

        p = subprocess.run(cmd, cwd=str(supervised_dir), capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(
                "predict 失敗\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout:\n{p.stdout}\n"
                f"stderr:\n{p.stderr}"
            )

        result_csv = supervised_dir / "results" / model_id / "prediction_result.csv"
        if not result_csv.is_file():
            raise FileNotFoundError(f"找不到預測輸出: {result_csv}")

        pred_df = pd.read_csv(result_csv)
        if pred_df.empty:
            raise ValueError(f"預測輸出為空: {result_csv}")
        pred_last = pred_df.iloc[-1].to_dict()

        return {
            "model_id": model_id,
            "input_rows": len(df_for_predict),
            "temp_csv": str(temp_csv),
            "result_csv": str(result_csv),
            "last_prediction": pred_last,
            "stdout_tail": "\n".join((p.stdout or "").splitlines()[-20:]),
        }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="測試 Selector（create_dataset / predict）")
    parser.add_argument(
        "--project-root",
        default=None,
        help="專案根目錄（預設為 selector.py 上一層，即 Trading_Framework）",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser(
        "create-dataset",
        help="建立資料：回傳 pred 那筆之前的 seq_len（不含 pred_date/session；不存檔）",
    )
    p_create.add_argument("--seq-len", type=int, default=128, help="輸入序列長度（歷史筆數）")
    p_create.add_argument(
        "--pred-date",
        default=None,
        help="可選；預測錨點日 YYYY-MM-DD。若不給，維持舊行為（取最後 seq_len 列）",
    )
    p_create.add_argument(
        "--session",
        default="night",
        help="盤別（type）：0/1 或 night/day",
    )

    p_pred = sub.add_parser("predict", help="使用指定 model_id 做預測")
    p_pred.add_argument(
        "--model-id",
        "-i",
        default="MTX_spot_separate_128_embed",
        help="checkpoint 資料夾名稱，例如 MTX_spot_separate_128_embed",
    )
    p_pred.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="可選；不給會從 model-id 解析（例如 ..._128_embed -> 128）",
    )
    p_pred.add_argument(
        "--pred-date",
        default=None,
        help="預測錨點日 YYYY-MM-DD；省略則用資料最後一個交易日",
    )
    p_pred.add_argument(
        "--session",
        default="night",
        help="盤別（type）：0/1 或 night/day",
    )

    args = parser.parse_args(argv)
    s = Selector(project_root=args.project_root)

    if args.cmd == "create-dataset":
        df = s.create_dataset(seq_len=args.seq_len, pred_date=args.pred_date, session=args.session)
        print(f"rows={len(df)}")
        if not df.empty:
            print(df.tail(10).to_string(index=False))
        return 0

    if args.cmd == "predict":
        res = s.predict(model_id=args.model_id, seq_len=args.seq_len, pred_date=args.pred_date, session=args.session)
        print("model_id:", res["model_id"])
        print("input_rows:", res["input_rows"])
        print("result_csv:", res["result_csv"])
        print("last_prediction:", res["last_prediction"])
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

