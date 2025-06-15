# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Text, Union
import copy
import math
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP

# qrun examples/benchmarks/Transformer/workflow_config_transformer_Alpha360.yaml ”


class TransformerModel(Model):
    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 512,      # <-- add this parameter
        dropout: float = 0,
        batch_size: int = 2048,
        n_epochs=100,
        lr=0.0001,
        metric="",
        early_stop=2**10,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # set hyper-parameters.
        self.d_model = d_model
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward  # <-- record for use
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.logger = get_module_logger("TransformerModel")
        self.logger.info("Naive Transformer:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # build the Transformer wrapper (pass FFN hidden size and device)
        self.model = Transformer(
            d_feat=d_feat,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            device=self.device,
            batch_first=True,  # optional: enable batch_first for efficiency
        )
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        # 支持单任务或双任务预测
        if pred.dim() == 1 or (pred.dim() == 2 and pred.size(1) == 1):
            # 只有一个输出（return）
            return F.mse_loss(pred.view(-1), label.view(-1))
        elif pred.dim() == 2 and pred.size(1) == 2:
            # 双头输出：return & volatility
            ret_pred, vol_pred = pred[:, 0], pred[:, 1]
            # label 可能是一维（只含 return）或二维
            if label.dim() == 1:
                ret_label = label
                vol_label = torch.zeros_like(ret_label)
            else:
                ret_label, vol_label = label[:, 0], label[:, 1]
            loss_ret = F.mse_loss(ret_pred, ret_label)
            loss_vol = F.mse_loss(vol_pred, vol_label)
            return loss_ret + 0.5 * loss_vol
        else:
            raise ValueError(f"Unsupported pred shape: {pred.shape}")

    def metric_fn(self, pred: torch.Tensor, label: torch.Tensor):
        # 處理雙頭輸出：只用第一個頭（return）來計算 metric
        if pred.dim() == 2 and pred.size(1) == 2:
            pred = pred[:, 0]  # 只取 return 預測，忽略 volatility
        
        # 確保 pred 和 label 都是一維
        pred_flat = pred.view(-1)
        label_flat = label.view(-1)
        
        # 檢查有效數據
        mask = torch.isfinite(label_flat) & torch.isfinite(pred_flat)
        
        if mask.sum() == 0:
            return 0.0
            
        pred_valid = pred_flat[mask]
        label_valid = label_flat[mask]

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred_valid, label_valid)
        elif self.metric == "ic":
            # Information Coefficient (Pearson correlation)
            if len(pred_valid) < 2:
                return 0.0
                
            pred_mean = pred_valid.mean()
            label_mean = label_valid.mean()
            
            cov = ((pred_valid - pred_mean) * (label_valid - label_mean)).mean()
            pred_std = pred_valid.std()
            label_std = label_valid.std()
            
            if pred_std == 0 or label_std == 0:
                return 0.0
            
            ic = cov / (pred_std * label_std)
            return float(ic)
        else:
            return -self.loss_fn(pred_valid, label_valid).item()

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.model.eval()

        scores = []
        losses = []
        all_pred = []
        all_label = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            with torch.no_grad():
                pred = self.model(feature)
                loss = self.loss_fn(pred, label)
                loss = torch.nan_to_num(loss, nan=1e3, posinf=1e3, neginf=-1e3)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(float(score))
                
                # 收集預測和標籤用於最終計算
                all_pred.append(pred.detach().cpu())
                all_label.append(label.detach().cpu())

        # 检查 all_pred 是否为空，避免 torch.cat 错误
        if len(all_pred) == 0:
            self.logger.warning("No valid batches found in test_epoch, returning default values")
            return 1e3, 0.0

        # 计算整体 Information Coefficient
        preds_cat = torch.cat(all_pred)  # [N, 2] 或 [N, 1]
        labels_cat = torch.cat(all_label)  # [N] 或 [N, 1]
        
        # 只用 return 頭計算最終 metric
        if preds_cat.dim() == 2 and preds_cat.size(1) == 2:
            pred_for_metric = preds_cat[:, 0]  # 只取 return 預測
        else:
            pred_for_metric = preds_cat.view(-1)
            
        label_for_metric = labels_cat.view(-1)
        ic_score = self.metric_fn(pred_for_metric, label_for_metric)

        # 返回平均 loss 和整體 IC
        if len(losses) == 0:
            avg_loss = 1e3
        else:
            avg_loss = float(torch.tensor(losses).mean())
        return avg_loss, float(ic_score)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            # test_epoch 现在只返回 (loss, score)
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

            with torch.no_grad():
                pred = self.model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(
        self,
        d_feat: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        device: torch.device,
        batch_first: bool = False,
    ):
        super().__init__()
        # record input feature dim for reshape in forward
        self.d_feat = d_feat
        # map raw features [*, F] -> model dimension [*, d_model]
        self.feature_layer = nn.Linear(d_feat, d_model)
        # positional encoding for Transformer
        from .pytorch_transformer import PositionalEncoding
        self.pos_encoder = PositionalEncoding(d_model)

        # build Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # decoder heads: return + volatility
        self.head_return = nn.Linear(d_model, 1)
        self.head_vol    = nn.Linear(d_model, 1)

        self.device = device
        self.to(device)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # ensure src has shape [B, F, T]
        if src.dim() == 2:
            # single time step case: [B, F] → [B, F, 1]
            src = src.unsqueeze(2)
        elif src.dim() != 3:
            raise ValueError(f"Transformer.forward expected src dim=2 or 3, got {src.dim()}")
        # src: [B, F, T] → [B, T, F]
        x = src.permute(0, 2, 1)              # [B,T,F]
        x = self.pos_encoder(self.feature_layer(x))  # [B,T,d_model]
        out = self.transformer_encoder(x)     # [B,T,d_model]
        last = out[:, -1, :]                  # [B,d_model]
        # 两个头分别算收益与波动
        ret  = self.head_return(last)         # [B,1]
        vol  = self.head_vol(last).abs()      # [B,1] 取正
        return torch.cat([ret, vol], dim=1)   # [B,2]
