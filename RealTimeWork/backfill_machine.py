# -*- coding: utf-8 -*-
"""
Backfill_machine：行為對齊 RealTimeWork/README.md（get_last_date、update_data）。

命令列測試（在 RealTimeWork 目錄下）：
  python backfill_machine.py last
  python backfill_machine.py update --contract 202503 --session day --session-date 2025-03-20
  python backfill_machine.py update -c 202503 -s day -d 2025-03-20
  python backfill_machine.py update -c 202503 -s day -d 2025-03-20 --last-session
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

try:
    import shioaji as sj
except ImportError:
    sj = None  # type: ignore


Session = Literal["day", "night"]


def _last_row_first_cell_raw(path: Path) -> Optional[str]:
    """檔尾最後一筆資料列的第一欄字串（MTX_1min 即完整 datetime）。"""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    size = path.stat().st_size
    chunk = min(size, 65_536)
    with open(path, "rb") as f:
        f.seek(max(0, size - chunk))
        if size > chunk:
            f.readline()
        tail = f.read().decode("utf-8", errors="replace")
    last_val: Optional[str] = None
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        first = line.split(",", 1)[0].strip().strip('"')
        if not first or first.lower() == "datetime":
            continue
        last_val = first
    return last_val


def _filter_trading_minutes(df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
    """移除非交易空檔：05:00–08:45、13:45–15:00（約 12 行邏輯，與常見 MTX 前處理一致）。"""
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col])
    t = out[dt_col].dt.time
    remove_1 = (t > pd.Timestamp("05:00:00").time()) & (t < pd.Timestamp("08:45:00").time())
    remove_2 = (t > pd.Timestamp("13:45:00").time()) & (t < pd.Timestamp("15:00:00").time())
    return out.loc[~(remove_1 | remove_2)].copy()


def _session_upper_bound(session: Session, session_date: str) -> pd.Timestamp:
    d = pd.Timestamp(session_date).normalize()
    if session == "night":
        return d.replace(hour=5, minute=0, second=0, microsecond=0)
    return d.replace(hour=13, minute=45, second=0, microsecond=0)


def _resample_15min(df_1m: pd.DataFrame, contract_month: str) -> pd.DataFrame:
    df = df_1m.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    out = (
        df.resample("15min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    out["contract_month"] = contract_month
    return out


def _append_mtx_daily_sep_from_new_15m(
    path_out: Path,
    df_15_new: pd.DataFrame,
    session: Session,
    session_date: str,
    contract_month: str,
) -> None:
    """
    本次 backfill 只新增一盤：用本批 15m K 聚成 OHLCV 一列，
    date = session_date，session = day|night，以 append 附加至 mtx_daily_sep（不刪舊列）。
    """
    if df_15_new is None or df_15_new.empty:
        print("本批無 15m 資料，略過 mtx_daily_sep。")
        return
    chunk = df_15_new.copy()
    chunk["datetime"] = pd.to_datetime(chunk["datetime"])
    chunk = chunk.sort_values("datetime")
    cm = str(contract_month).strip()
    if "contract_month" in chunk.columns and chunk["contract_month"].notna().any():
        cm = str(chunk["contract_month"].iloc[-1]).strip() or cm
    row = {
        "date": pd.Timestamp(session_date).strftime("%Y-%m-%d"),
        "session": session,
        "open": float(chunk["open"].iloc[0]),
        "high": float(chunk["high"].max()),
        "low": float(chunk["low"].min()),
        "close": float(chunk["close"].iloc[-1]),
        "volume": float(chunk["volume"].sum()),
        "contract_month": cm,
    }
    cols = ["date", "session", "open", "high", "low", "close", "volume", "contract_month"]
    one = pd.DataFrame([row])[cols]
    path_out.parent.mkdir(parents=True, exist_ok=True)
    header = not path_out.is_file() or path_out.stat().st_size == 0
    one.to_csv(path_out, index=False, mode="a", header=header)
    print(f"mtx_daily_sep: append ({row['date']}, {row['session']}) -> {path_out}")


class Backfill_machine:
    """
    README：
      - get_last_date：MTX_1min 最後一列的 datetime
      - update_data：外盤近 10 日合併去重；MTX 1m／15m；append 15m 後依 session_date+session append mtx_daily_sep 一列
    """

    def __init__(
        self,
        project_root: Optional[str | Path] = None,
        raw_data_dir: Optional[str | Path] = None,
        api_config: Optional[str | Path] = None,
        shioaji_symbol: str = "MXF",
    ) -> None:
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent.parent
        self.raw_data_dir = Path(raw_data_dir).resolve() if raw_data_dir else self.project_root / "MTX_rawdata"
        self.api_config = Path(api_config).resolve() if api_config else self.project_root / "api_key.yaml"
        self.shioaji_symbol = shioaji_symbol

        self._path_1m = self.raw_data_dir / "MTX_1min.csv"
        self._path_15 = self.raw_data_dir / "MTX_15min.csv"
        self._path_mtx_daily_sep = self.raw_data_dir / "mtx_daily_sep.csv"
        self._paths_index = {
            "sp500": self.raw_data_dir / "SP500_daily.csv",
            "n225": self.raw_data_dir / "N225_daily.csv",
            "twse": self.raw_data_dir / "TWSE_daily.csv",
        }
        self._tickers = {"sp500": "^GSPC", "n225": "^N225", "twse": "^TWII"}

    def get_last_date(self) -> Optional[str]:
        """MTX_1min 最後一列的 datetime 字串；無檔／無資料列則 None。"""
        return _last_row_first_cell_raw(self._path_1m)

    def update_data(
        self,
        contract: str,
        session: Session,
        session_date: str,
        *,
        last_session: bool = False,
    ) -> None:
        """
        SP500／N225／TWSE：抓最近 10 個曆日區間與原檔合併、依 date 去重。
        MTX_1min：Shioaji kbars，start = get_last_date 的日期（無則用 session_date），
        API end 曆日 = session_date 次日；再依 README 盤別上界與非交易時段 filter，且嚴格大於 get_last_date。
        最後將本次新增之 1m 聚合成 15m 附加至 MTX_15min，並用本批 15m + session_date + session append mtx_daily_sep 一列。
        last_session=True：append mtx_daily_sep 時去掉本批 df_15 最後一根 15m（結算棒不用）；MTX_15min 仍寫完整 df_15。
        """
        self._update_indices_latest_10d()
        self._update_mtx_1m_and_15m(
            contract=str(contract).strip(),
            session=session,
            session_date=session_date,
            last_session=last_session,
        )

    def _update_indices_latest_10d(self) -> None:
        import yfinance as yf

        today = pd.Timestamp.now().normalize()
        dl_start = today - pd.Timedelta(days=10)
        end = today

        for name, path in self._paths_index.items():
            ticker = self._tickers[name]
            t = yf.Ticker(ticker)
            hist = t.history(
                start=dl_start.strftime("%Y-%m-%d"),
                end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
            )
            if hist is None or hist.empty:
                continue
            hist = hist.reset_index()
            date_col = "Date" if "Date" in hist.columns else hist.columns[0]
            close_col = "Close" if "Close" in hist.columns else None
            if close_col is None:
                continue
            new_df = hist[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})
            new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            new_df = new_df.dropna(subset=["date", "close"])

            if path.is_file() and path.stat().st_size > 0:
                old = pd.read_csv(path)
                if "date" not in old.columns or "close" not in old.columns:
                    raise ValueError(f"{path} 需有 date, close 欄")
                old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                old = old.dropna(subset=["date"])
                merged = pd.concat([old, new_df], ignore_index=True)
            else:
                merged = new_df

            merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(path, index=False)
            print(f"[{name}] merged rows={len(merged)} -> {path}")

    def _update_mtx_1m_and_15m(
        self,
        contract: str,
        session: Session,
        session_date: str,
        *,
        last_session: bool = False,
    ) -> None:
        if sj is None:
            raise ImportError("請安裝 shioaji")
        if yaml is None:
            raise ImportError("請安裝 PyYAML")

        last_s = self.get_last_date()
        if last_s:
            last_ts = pd.to_datetime(last_s)
            api_start = last_ts.normalize().strftime("%Y-%m-%d")
        else:
            last_ts = None
            api_start = pd.Timestamp(session_date).strftime("%Y-%m-%d")

        api_end = (pd.Timestamp(session_date).normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if not self.api_config.is_file():
            raise FileNotFoundError(f"找不到 API 設定: {self.api_config}")
        cfg = yaml.safe_load(self.api_config.read_text(encoding="utf-8"))
        pid = cfg["Future"]["api_key"]
        pwd = cfg["Future"]["secret_key"]
        cert_name = cfg["Future"]["cert"]
        ca_pass = cfg["Future"]["passwd"]
        ca_path = self.project_root / "cert" / f"{cert_name}.pfx"
        if not ca_path.is_file():
            raise FileNotFoundError(f"找不到憑證: {ca_path}")

        api = sj.Shioaji(simulation=False)
        api.login(pid, pwd)
        api.activate_ca(ca_path=str(ca_path), ca_passwd=ca_pass)

        try:
            sym = self.shioaji_symbol
            fut = api.Contracts.Futures[sym][f"{sym}R1"]
            kbars = api.kbars(contract=fut, start=api_start, end=api_end)
            df = pd.DataFrame({**kbars})
        finally:
            try:
                api.logout()
            except Exception:
                pass

        if df.empty:
            print("Shioaji kbars 空，未更新 MTX_1min。")
            return

        df["ts"] = pd.to_datetime(df["ts"])
        df = df.rename(
            columns={
                "ts": "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df[["datetime", "open", "high", "low", "close", "volume"]]

        hi = _session_upper_bound(session, session_date)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.loc[df["datetime"] <= hi].copy()

        df = _filter_trading_minutes(df, "datetime")

        if last_ts is not None:
            df = df.loc[df["datetime"] > last_ts].copy()

        if df.empty:
            print("篩選後無新 1m 資料，未寫入。")
            return

        df = df.sort_values("datetime")
        df_15 = _resample_15min(df, contract)
        df_15_mtx = df_15.copy()
        if last_session and not df_15_mtx.empty:
            df_15_mtx = df_15_mtx.iloc[:-1].copy()

        out_1m = df.copy()
        out_1m["datetime"] = out_1m["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        self._path_1m.parent.mkdir(parents=True, exist_ok=True)
        header = not self._path_1m.is_file() or self._path_1m.stat().st_size == 0
        out_1m.to_csv(self._path_1m, index=False, mode="a", header=header)
        print(f"MTX_1min appended {len(out_1m)} rows")

        _append_mtx_daily_sep_from_new_15m(
            self._path_mtx_daily_sep,
            df_15_mtx,
            session=session,
            session_date=session_date,
            contract_month=contract,
        )

        df_15["datetime"] = pd.to_datetime(df_15["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_15 = df_15.sort_values("datetime")

        self._path_15.parent.mkdir(parents=True, exist_ok=True)
        h15 = not self._path_15.is_file() or self._path_15.stat().st_size == 0
        df_15.to_csv(self._path_15, index=False, mode="a", header=h15)
        print(f"MTX_15min appended {len(df_15)} rows")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="測試 Backfill_machine（get_last_date / update_data）")
    parser.add_argument(
        "-p",
        "--project-root",
        default=None,
        help="專案根目錄（預設：本檔上一層，即 Trading_Framework）",
    )
    parser.add_argument(
        "-r",
        "--raw-dir",
        default=None,
        help="MTX_rawdata 目錄（預設：<project-root>/MTX_rawdata）",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_last = sub.add_parser("last", help="印出 MTX_1min 最後一列 datetime（get_last_date）")

    p_up = sub.add_parser("update", help="執行 update_data（外盤合併 + Shioaji 補 1m/15m）")
    p_up.add_argument("-c", "--contract", required=True, help="契約月份，例 202503")
    p_up.add_argument("-s", "--session", required=True, choices=["day", "night"], help="日盤 / 夜盤")
    p_up.add_argument(
        "-d",
        "--session-date",
        required=True,
        metavar="DATE",
        help="交易日 YYYY-MM-DD（夜盤為 05:00 所在曆日）",
    )
    p_up.add_argument(
        "--last-session",
        action="store_true",
        help="mtx_daily_sep 聚合前去掉本批 15m 最後一根；MTX_15min 仍寫入完整 15m",
    )

    args = parser.parse_args(argv)

    bf = Backfill_machine(project_root=args.project_root, raw_data_dir=args.raw_dir)

    if args.cmd == "last":
        v = bf.get_last_date()
        print(v if v is not None else "(無資料或無檔案)")
        return 0

    if args.cmd == "update":
        bf.update_data(
            contract=args.contract,
            session=args.session,
            session_date=args.session_date,
            last_session=bool(args.last_session),
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
