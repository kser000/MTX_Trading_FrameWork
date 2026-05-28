from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import pandas as pd

import os
import pickle
import time

import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class MSEDirectionLoss(nn.Module):
    """
    MSE + 方向懲罰：預測與真實同向時不額外懲罰，反向時加上 max(0, -pred*true)。
    適合預測 log_return 時同時顧及漲跌方向。
    """
    def __init__(self, direction_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.direction_weight = direction_weight

    def forward(self, pred, true):
        mse = self.mse(pred, true)
        # 同向: pred*true > 0 -> -pred*true < 0 -> clamp 為 0；反向: pred*true < 0 -> 懲罰
        direction_penalty = torch.clamp(-pred * true, min=0.0).mean()
        return mse + self.direction_weight * direction_penalty


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        if self.args.model != 'PatchTST':
            raise ValueError(f"此 Exp_Main 僅支援 PatchTST，目前 --model={self.args.model!r}")
        model = PatchTST.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        loss_name = (getattr(self.args, 'loss', None) or 'mse').lower()
        if loss_name == 'mse':
            return nn.MSELoss()
        if loss_name == 'mse_direction':
            direction_weight = getattr(self.args, 'loss_direction_weight', 1.0)
            return MSEDirectionLoss(direction_weight=direction_weight)
        return nn.MSELoss()

    def _unpack_batch(self, batch):
        """Unpack batch (4 or 5 elements); 5th is session_id for PatchTST session embedding."""
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch[0], batch[1], batch[2], batch[3]
        batch_session_id = None
        if len(batch) == 5 and getattr(self.args, 'use_session_embed', 0):
            batch_session_id = batch[4].long().to(self.device)
        return batch_x, batch_y, batch_x_mark, batch_y_mark, batch_session_id

    def _patchtst_forward(self, batch_x, batch_session_id):
        """PatchTST：只吃 encoder 輸入；可選 session embedding。"""
        if batch_session_id is not None:
            return self.model(batch_x, session_id=batch_session_id)
        return self.model(batch_x)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, _, _, batch_session_id = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._patchtst_forward(batch_x, batch_session_id)
                else:
                    outputs = self._patchtst_forward(batch_x, batch_session_id)
                outputs = outputs[:, -1:, :]
                batch_y = batch_y[:, -1:, :].to(self.device)
                if self.args.features == 'MS':
                    tci = getattr(vali_data, 'target_col_idx', 0)
                    outputs = outputs[:, :, tci:tci + 1]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 將 train 的 scaler 存到 checkpoint 資料夾，供 predict 使用
        if hasattr(train_data, 'scaler'):
            scaler_path = os.path.join(path, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(train_data.scaler, f)
            print('Train scaler saved: {}'.format(scaler_path))

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                batch_x, batch_y, _, _, batch_session_id = self._unpack_batch(batch)
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._patchtst_forward(batch_x, batch_session_id)
                        outputs = outputs[:, -1:, :]
                        batch_y = batch_y[:, -1:, :].to(self.device)
                        if self.args.features == 'MS':
                            tci = getattr(train_data, 'target_col_idx', 0)
                            outputs = outputs[:, :, tci:tci + 1]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self._patchtst_forward(batch_x, batch_session_id)
                    outputs = outputs[:, -1:, :]
                    batch_y = batch_y[:, -1:, :].to(self.device)
                    if self.args.features == 'MS':
                        tci = getattr(train_data, 'target_col_idx', 0)
                        outputs = outputs[:, :, tci:tci + 1]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 將 train 的 scaler 存到 checkpoint 資料夾，供 predict 使用
        if hasattr(test_data, 'scaler'):
            scaler_path = os.path.join(path, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(test_data.scaler, f)
            print('Train scaler saved: {}'.format(scaler_path))

        preds = []
        trues = []
        inputx = []
        date_list = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        seq_len = self.args.seq_len
        batch_size = self.args.batch_size
        # When `drop_last=False`, the last batch can be smaller.
        # Use a running cursor of processed samples to align `date_list` correctly.
        sample_cursor = 0

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, _, _, batch_session_id = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # 與 vali / predict 一致：use_amp 時 forward 走 autocast，否則 test 與 predict 的 pred 會不一致
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._patchtst_forward(batch_x, batch_session_id)
                else:
                    outputs = self._patchtst_forward(batch_x, batch_session_id)
                outputs = outputs[:, -1:, :]
                batch_y = batch_y[:, -1:, :].to(self.device)
                if self.args.features == 'MS':
                    tci = getattr(test_data, 'target_col_idx', 0)
                    outputs = outputs[:, :, tci:tci + 1]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                # 每個 sample 對應預測錨點日（單步 horizon）
                n_samples = pred.shape[0]
                for k in range(n_samples):
                    idx = sample_cursor + k + seq_len
                    if idx < len(test_data.dates):
                        date_list.append(test_data.dates[idx])
                if i % 20 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    tci_v = test_data.target_col_idx
                    gt = np.concatenate((input_np[0, :, tci_v], true[0, :, 0]), axis=0)
                    pred_v = np.concatenate((input_np[0, :, tci_v], pred[0, :, 0]), axis=0)
                    visual(gt, pred_v, os.path.join(folder_path, str(i) + '.pdf'))
                sample_cursor += n_samples

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        # When drop_last=False, the last batch can be smaller, so each element in
        # `preds/trues/inputx` may have different batch dimension. Using np.array()
        # will fail with "inhomogeneous shape"; concatenate fixes this.
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)

        # 輸出 CSV：date, log_return_pred, log_return_true, direction_pred, direction_true（還原 scale）
        tci = test_data.target_col_idx
        if hasattr(test_data, 'scaler') and hasattr(test_data.scaler, 'scale_') and len(date_list) == preds.size:
            try:
                # 還原 target 欄（用 target_col_idx）
                scale_val = float(test_data.scaler.scale_[tci])
                mean_val = float(test_data.scaler.mean_[tci])
                pred_flat = preds.reshape(-1, preds.shape[-1])
                true_flat = trues.reshape(-1, trues.shape[-1])
                log_return_pred = pred_flat[:, 0] * scale_val + mean_val
                log_return_true = true_flat[:, 0] * scale_val + mean_val
            except Exception:
                log_return_pred = preds.reshape(-1, preds.shape[-1])[:, 0]
                log_return_true = trues.reshape(-1, trues.shape[-1])[:, 0]
        else:
            log_return_pred = preds.reshape(-1, preds.shape[-1])[:, 0]
            log_return_true = trues.reshape(-1, trues.shape[-1])[:, 0]
        direction_pred = (log_return_pred > 0).astype(int)
        direction_true = (log_return_true > 0).astype(int)
        n_rows = len(log_return_pred)
        dates_out = date_list[:n_rows] if len(date_list) >= n_rows else [None] * n_rows
        df_out = pd.DataFrame({
            'date': dates_out,
            'log_return_pred': log_return_pred,
            'log_return_true': log_return_true,
            'direction_pred': direction_pred,
            'direction_true': direction_true,
        })
        csv_path = os.path.join(folder_path, 'test_result.csv')
        df_out.to_csv(csv_path, index=False)
        print('Test result CSV saved: {}'.format(csv_path))

        return

    def predict(self, setting, load=False):
        """
        依 pred_start / pred_end 做多日預測（或單筆）：載入 checkpoint，遍歷 pred dataset，
        每筆為該預測日之前 seq_len 的輸入，輸出 results/<setting>/prediction_result.csv
        （date, type, log_return_pred, log_return_true, direction_pred, direction_true；多日模式含真值）。
        """
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            if not os.path.exists(best_model_path):
                raise FileNotFoundError('Checkpoint not found: {}'.format(best_model_path))
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print('Loaded checkpoint: {}'.format(best_model_path))

        pred_data, pred_loader = self._get_data(flag='pred')
        if len(pred_loader) == 0:
            raise ValueError('Prediction dataset is empty (no batch).')

        pred_dates = getattr(pred_data, 'pred_dates_list', None)
        pred_types = getattr(pred_data, 'pred_type_list', None)  # 每 session 一筆：0=夜盤 1=日盤
        tci = getattr(pred_data, 'target_col_idx', 0)
        scale_val = mean_val = None
        if hasattr(pred_data, 'scaler') and hasattr(pred_data.scaler, 'scale_'):
            scale_val = float(pred_data.scaler.scale_[tci])
            mean_val = float(pred_data.scaler.mean_[tci])

        all_pred_raw = []
        all_dates = []
        gt_scaled_list = getattr(pred_data, 'pred_ground_truth_scaled', None)
        all_true_raw = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(pred_loader):
                batch_x, _, _, _, batch_session_id = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._patchtst_forward(batch_x, batch_session_id)
                else:
                    outputs = self._patchtst_forward(batch_x, batch_session_id)
                outputs = outputs[:, -1:, :]
                if self.args.features == 'MS':
                    outputs = outputs[:, :, tci:tci + 1]
                pred_raw = outputs.detach().cpu().numpy()
                all_pred_raw.append(pred_raw)
                # pred_loader 在 data_factory 中固定 batch_size=1
                base = batch_idx

                if pred_dates is not None and base < len(pred_dates):
                    all_dates.append(pred_dates[base])

                if gt_scaled_list is not None:
                    if base < len(gt_scaled_list):
                        gt_val = np.asarray(gt_scaled_list[base], dtype=np.float64)
                    else:
                        gt_val = np.asarray(np.nan, dtype=np.float64)
                    all_true_raw.append(gt_val)

                # 每筆（batch=1）立刻印出 unscaled log_return_true / log_return_pred
                pred_scaled_val = float(pred_raw.reshape(-1)[0])
                if scale_val is not None and mean_val is not None:
                    pred_log_return_pred = pred_scaled_val * scale_val + mean_val
                else:
                    pred_log_return_pred = pred_scaled_val

                pred_log_return_true = float('nan')
                if gt_scaled_list is not None and base < len(gt_scaled_list):
                    true_scaled_val = float(np.asarray(gt_scaled_list[base], dtype=np.float64))
                    if scale_val is not None and mean_val is not None:
                        pred_log_return_true = true_scaled_val * scale_val + mean_val
                    else:
                        pred_log_return_true = true_scaled_val

                print(f"[predict] idx={base} log_return_pred={pred_log_return_pred} log_return_true={pred_log_return_true}")

        pred_raw = np.concatenate(all_pred_raw, axis=0)
        log_return_flat = pred_raw.reshape(-1)
        if scale_val is not None and mean_val is not None:
            log_return_flat = log_return_flat * scale_val + mean_val
        n_samples = pred_raw.shape[0]
        log_return_list = log_return_flat.tolist()
        direction_list = [1 if (float(lr) > 0) else 0 for lr in log_return_list]

        # ground truth（與 test_result 相同：還原 scale 後的 log_return_true / direction_true）
        if all_true_raw and len(all_true_raw) == n_samples:
            true_scaled = np.stack(all_true_raw, axis=0)
            if scale_val is not None and mean_val is not None:
                true_flat = true_scaled.reshape(-1) * scale_val + mean_val
            else:
                true_flat = true_scaled.reshape(-1)
            log_return_true_list = true_flat.tolist()

            def _direction_from_log_return(lr):
                a = np.asarray(lr if isinstance(lr, list) else [lr], dtype=np.float64)
                if not np.isfinite(a).any():
                    return 0
                return int(np.nanmean(a) > 0)

            direction_true_list = [_direction_from_log_return(lr) for lr in log_return_true_list]
        else:
            log_return_true_list = [float('nan')] * n_samples
            direction_true_list = [0] * n_samples

        if all_dates and len(all_dates) == len(log_return_list):
            dates_out = all_dates
        else:
            dates_out = [None] * len(log_return_list)
        if pred_types is not None and len(pred_types) == len(log_return_list):
            types_out = pred_types
        else:
            types_out = [None] * len(log_return_list)

        df_out = pd.DataFrame({
            'date': dates_out,
            'type': types_out,
            'log_return_pred': log_return_list,
            'log_return_true': log_return_true_list,
            'direction_pred': direction_list,
            'direction_true': direction_true_list,
        })

        # 單日預測（n_samples == 1）只回傳結果、不寫檔；多日才存 csv / npy
        if n_samples <= 1:
            print('[predict] single-day mode (n_samples={}): skip saving npy / csv'.format(n_samples))
        else:
            folder_path = './results/' + setting + '/'
            os.makedirs(folder_path, exist_ok=True)
            np.save(os.path.join(folder_path, 'real_prediction.npy'), pred_raw)
            if all_true_raw and len(all_true_raw) == n_samples:
                np.save(os.path.join(folder_path, 'pred_ground_truth_scaled.npy'), np.stack(all_true_raw, axis=0))
            csv_path = os.path.join(folder_path, 'prediction_result.csv')
            df_out.to_csv(csv_path, index=False)
            print('Prediction saved: {} ({} rows)'.format(csv_path, len(df_out)))

        return {
            'log_return_pred': log_return_list,
            'log_return_true': log_return_true_list,
            'direction': direction_list,
            'direction_true': direction_true_list,
            'pred_raw': pred_raw,
            'date': dates_out,
        }