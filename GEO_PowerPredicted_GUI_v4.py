import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import onnxruntime as ort
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# ===============================
# UI 配色
# ===============================
BG = "#1e1e2f"      # 背景
CARD = "#2b2b3d"    # 卡片
TEXT = "#e5e7eb"    # 文字
ACCENT = "#3b82f6"  # 按钮 / 高亮
# ===============================
# ONNX 模型加载
# ===============================
def get_model_path():
    return "AWPL_checkpoints_300_900_30_Fusion_W2_model.onnx"

sess = ort.InferenceSession(get_model_path())

# ===============================
# ERA 数据加载（完全复刻你原逻辑）
# ===============================
ERA_FILES = [
    "era5_pressure_merged_1hPa_filtered_normalized",
    "era5_pressure_merged_5hPa_filtered_normalized",
    "era5_pressure_merged_30hPa_filtered_normalized",
    "era5_pressure_merged_200hPa_filtered_normalized",
    "era5_pressure_merged_650hPa_filtered_normalized",
    "era5_surface_merged_all_filter_normalized",
]

SPLIT_DIR = "./splits"

def load_era_dict(split):
    dfs = []
    for name in ERA_FILES:
        df = pd.read_csv(
            f"{SPLIT_DIR}/{name}_{split}.csv",
            parse_dates=["valid_time"]
        )
        df = df.set_index(df["valid_time"].dt.floor("h"))
        df = df.drop(columns=["valid_time"])
        dfs.append(df)

    era = pd.concat(dfs, axis=1)

    era_dict = {
        hour: row.to_numpy(dtype=np.float32)
        for hour, row in era.iterrows()
    }
    return era_dict

# ===============================
# 数据集（精简版，只保留推理必要部分）
# ===============================
class PowerDataset:
    def __init__(self, split, Y, P, rate):
        self.rate = rate
        self.Y = Y // rate
        self.P = P // rate

        self.df = pd.read_csv(
            f"{SPLIT_DIR}/power_{split}.csv",
            parse_dates=["Timestamp"]
        ).sort_values("Timestamp")

        self.df["hour"] = self.df["Timestamp"].dt.floor("h")

        self.era_dict = load_era_dict(split)

    def generate(self):
        for hour, g in self.df.groupby("hour"):
            if hour not in self.era_dict:
                continue

            power_raw = g["TimePower"].to_numpy(dtype=np.float32)
            times_raw = g["Timestamp"].to_numpy()

            n_points = len(power_raw) // self.rate
            if n_points < (self.Y + self.P):
                continue

            power = power_raw[:n_points * self.rate].reshape(-1, self.rate).mean(axis=1)
            times = times_raw[:n_points * self.rate:self.rate]

            atm = self.era_dict[hour]

            for i in range(len(power) - self.Y - self.P + 1):
                x = power[i:i+self.Y]
                y = power[i+self.Y:i+self.Y+self.P]

                yield (
                    x, y, atm,
                    str(times[i])
                )

# ===============================
# 推理 + 可视化
# ===============================
def run_history():
    try:
        target_date = date_entry.get().strip()

        if not target_date:
            messagebox.showerror("错误", "请输入日期，例如 2025-12-12")
            return

        Yhis = 300
        Phis = 900
        rate = 30

        dataset = PowerDataset("test", Yhis, Phis, rate)

        preds_all = []
        trues_all = []

        found = False

        for x, y, a, t in dataset.generate():

            if not found:
                if target_date in t:
                    found = True
                else:
                    continue

            x = x.reshape(1, -1, 1).astype(np.float32)
            a = a.reshape(1, -1).astype(np.float32)
            atm_input = np.zeros((1, 241), dtype=np.float32)
            # ===== normalization =====
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-8
            x_norm = (x - mean) / std
            print(f"x_norm.shape:{x_norm.shape}")
            pred = sess.run(None, {
                "power_seq": x_norm,
                "atm": atm_input
            })[0]
            print(f"pred.shape:{pred.shape}")
            pred = pred.squeeze(-1)
            pred = pred * std.squeeze(-1) + mean.squeeze(-1)

            preds_all.append(pred.flatten())
            trues_all.append(y.flatten())
            print(f"preds_all.shape:{np.array(preds_all).shape}")
            if len(preds_all) > 50:
                break

        if not found:
            messagebox.showerror("错误", f"没有找到日期 {target_date}")
            return

        y_pred = np.concatenate(preds_all)
        y_true = np.concatenate(trues_all)
        y_pred = y_pred[:30]
        y_true = y_true[:30]
        plot_results(y_true, y_pred)

    except Exception as e:
        messagebox.showerror("错误", str(e))


# ===============================
# 绘图（论文级）
# ===============================
def plot_results(y_true, y_pred):

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # ===== CDF =====
    true_sorted = np.sort(y_true)
    pred_sorted = np.sort(y_pred)

    true_cdf = np.arange(len(true_sorted)) / len(true_sorted)
    pred_cdf = np.arange(len(pred_sorted)) / len(pred_sorted)

    x = np.linspace(
        min(true_sorted.min(), pred_sorted.min()),
        max(true_sorted.max(), pred_sorted.max()),
        1000
    )

    true_cdf_interp = np.interp(x, true_sorted, true_cdf)
    pred_cdf_interp = np.interp(x, pred_sorted, pred_cdf)

    ks = np.max(np.abs(true_cdf_interp - pred_cdf_interp))

    # ===== 清空并重画 =====
    fig.clear()

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(y_true[:1000], label="True")
    ax1.plot(y_pred[:1000], label="Pred")
    ax1.set_title(f"Time Series (RMSE={rmse:.4f})")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x, true_cdf_interp, label="True CDF")
    ax2.plot(x, pred_cdf_interp, label="Pred CDF")
    ax2.set_title(f"CDF (KS={ks:.4f})")
    ax2.legend()

    fig.tight_layout()

    # ✅ 刷新 GUI 里的画布
    canvas.draw()

    # ✅ 更新指标卡片
    rmse_label.config(text=f"{rmse:.4f}")
    ks_label.config(text=f"{ks:.4f}")
# def plot_results(y_true, y_pred):
#
#     rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
#
#     # ===== CDF =====
#     true_sorted = np.sort(y_true)
#     pred_sorted = np.sort(y_pred)
#
#     true_cdf = np.arange(len(true_sorted)) / len(true_sorted)
#     pred_cdf = np.arange(len(pred_sorted)) / len(pred_sorted)
#
#     x = np.linspace(
#         min(true_sorted.min(), pred_sorted.min()),
#         max(true_sorted.max(), pred_sorted.max()),
#         1000
#     )
#
#     true_cdf_interp = np.interp(x, true_sorted, true_cdf)
#     pred_cdf_interp = np.interp(x, pred_sorted, pred_cdf)
#
#     ks = np.max(np.abs(true_cdf_interp - pred_cdf_interp))
#
#     # ===== 画图 =====
#     plt.figure(figsize=(12, 5))
#
#     # 时序
#     plt.subplot(1, 2, 1)
#     plt.plot(y_true[:1000], label="True")
#     plt.plot(y_pred[:1000], label="Pred")
#     plt.title(f"Time Series (RMSE={rmse:.4f})")
#     plt.legend()
#
#     # CDF
#     plt.subplot(1, 2, 2)
#     plt.plot(x, true_cdf_interp, label="True CDF")
#     plt.plot(x, pred_cdf_interp, label="Pred CDF")
#     plt.title(f"CDF (KS={ks:.4f})")
#     plt.legend()
#
#     plt.tight_layout()
#
#
#     rmse_label.config(text=f"{rmse:.4f}")
#     mae_label.config(text="--")  # 你暂时没算 MAE
#     ks_label.config(text=f"{ks:.4f}")
#     canvas.draw()
#
# ===============================
# GUI（高级风）
# ===============================
# ===============================
# UI 界面（科研深色风）
# ===============================
root = tk.Tk()
root.title("Satellite Channel Prediction System")
root.geometry("1000x850")
root.configure(bg=BG)

# ---------- 标题 ----------
title = tk.Label(
    root,
    text="卫星信道功率预测系统（科研版）",
    font=("Microsoft YaHei", 18, "bold"),
    bg=BG,
    fg=TEXT
)
title.pack(pady=(15, 10))

# ---------- 控制面板 ----------
ctrl_frame = tk.Frame(root, bg=CARD, padx=20, pady=15)
ctrl_frame.pack(fill="x", padx=20)

tk.Label(
    ctrl_frame,
    text="历史数据日期 (YYYY-MM-DD):",
    bg=CARD,
    fg=TEXT,
    font=("Arial", 11)
).pack(side="left")

date_entry = tk.Entry(
    ctrl_frame,
    width=22,
    font=("Arial", 11),
    relief="flat",
    highlightthickness=1,
    highlightbackground=ACCENT,
    highlightcolor=ACCENT
)
date_entry.pack(side="left", padx=8)

tk.Button(
    ctrl_frame,
    text="运行预测",
    command=run_history,
    bg=ACCENT,
    fg=BG,
    font=("Arial", 11, "bold"),
    relief="flat",
    cursor="hand2"
).pack(side="right")

# ---------- 指标展示 ----------
metric_frame = tk.Frame(root, bg=BG, pady=10)
metric_frame.pack(fill="x", padx=20)

def create_metric(parent, label):
    f = tk.Frame(parent, bg=CARD, padx=15, pady=10)
    f.pack(side="left", expand=True, fill="x", padx=5)

    tk.Label(f, text=label, bg=CARD, fg=TEXT).pack()
    val = tk.Label(f, text="--", bg=CARD, fg=ACCENT, font=("Arial", 16, "bold"))
    val.pack()
    return val

rmse_label = create_metric(metric_frame, "RMSE")
mae_label = create_metric(metric_frame, "MAE")
ks_label = create_metric(metric_frame, "KS Statistic")

# ---------- Matplotlib 画布 ----------
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

fig = plt.figure(figsize=(8, 6), facecolor=BG)
plt.rcParams.update({
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.facecolor": CARD,
    "figure.facecolor": BG,
})

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=10)

root.mainloop()