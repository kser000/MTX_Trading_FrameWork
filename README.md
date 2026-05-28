# Trading Framework

本框架以 PatchTST / xLSTM 為「行情方向預測（Selector）」，配合 MTXEnv（Gymnasium 環境）與 PPO 強化學習做 MTX 期貨交易決策。

## 架構總覽

```
原始行情 (MTX 15min) ─┬─> PatchTST  ┐
                      └─> xLSTM     ├─> Selector：判斷下一段行情多 / 空 / 不交易
                                    │
                                    ▼
                    MTXEnv (Gymnasium) ─> Long / Short RL Trader (MaskablePPO)
                                                     │
                                                     ▼
                                       回測 (CombineTrader) / 即時下單 (RealTimeTrader)
```

## 模組簡述

### PatchTST (`PatchTST/`)
- Time-series Transformer，將時序切成 patch 後丟進 Transformer encoder 預測下一個 session 的 log return。
- 透過 `PatchTST/PatchTST_supervised/run_*.py` 訓練 / 測試 / 預測，輸出 `direction_pred`、`log_return_pred`，供 Selector 使用。

### xLSTM (`TraderSelector/`)
- 以 xLSTM_TS 為主幹，做日 / 季 / 半年滾動的訓練與微調，輸出方向分類（binary / 3-class）或 log return 回歸。
- 主要訓練腳本：`train_xLSTM*.py`，測試腳本：`test_xLSTM*.py`；資料前處理為 `TS_preprocess.py` / `regression_preprocess.py`。

### MTXEnv (`MTXEnv/`)
- 基於 Gymnasium 的 MTX 期貨交易環境，提供 RL 訓練 / 測試所需的 state、action（多 / 空 / 平倉）、reward 與部位 / 風控限制。
- 支援多種環境變體（`MTXEnvTrader`、`MTXEnvLong*`、`MTXEnvShort*`、`MTXEnvRealTimeTrader` …），分別給回測與即時交易使用。
- RL agent 採 `sb3_contrib.MaskablePPO`，搭配環境的 `valid_action_mask_*` 限制非法動作。

## 主要入口
- 訓練 RL：`STB3_Trainer.py`
- 回測 + Selector 整合：`CombineTrader.py`
- 即時下單（Shioaji）：`RealTimeTrader.py`
