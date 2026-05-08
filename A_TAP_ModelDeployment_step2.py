import onnxruntime as ort
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

def get_layer_dims(split_dir, era_files, split="train"):
    # 大气参数已经拼接了，为了方便分开，获取拼接前每个文件的参数个数
    dims = []
    for name in era_files:
        df = pd.read_csv(f"{split_dir}/{name}_{split}.csv", nrows=1)
        # 排除 valid_time 列
        cols = [c for c in df.columns if c != 'valid_time']
        dims.append(len(cols))
    return dims
def build_model(model_name,
                layer_dims,
                seq_len=300,
                P=300,
                atm_dim=None,
                device="cuda"):
    if atm_dim is None:
        atm_dim = sum(layer_dims)

    if model_name == "Fusion":  # Advanced Graph Transformer
        model = AdvancedSatelliteFusionModel(
            layer_dims=layer_dims,
            power_hidden=64,
            gnn_hidden=64,
            P=P,
            num_layers_atm=2
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model.to(device)
class AdvancedSatelliteFusionModel(nn.Module):
    def __init__(self,
                 layer_dims,
                 power_hidden=64,
                 gnn_hidden=64,
                 P=300,
                 num_layers_atm=2):
        super().__init__()

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.P = P

        # assert sum(layer_dims) > 0

        # =====================================================
        # 1️⃣ 异构大气层编码
        # =====================================================
        self.atm_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, gnn_hidden),
                nn.LayerNorm(gnn_hidden),
                nn.ReLU()
            ) for dim in layer_dims
        ])

        # =====================================================
        # 2️⃣ 垂直位置编码（物理高度感知）
        # =====================================================
        self.layer_pos = nn.Parameter(
            torch.randn(1, self.num_layers, gnn_hidden)
        )

        # =====================================================
        # 3️⃣ 垂直 Graph Transformer
        # =====================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gnn_hidden,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1
        )

        self.vertical_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers_atm
        )

        # =====================================================
        # 4️⃣ 功率时序分支
        # =====================================================
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=power_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # 时序注意力
        self.temporal_attn = nn.Sequential(
            nn.Linear(power_hidden, power_hidden),
            nn.Tanh(),
            nn.Linear(power_hidden, 1)
        )

        # =====================================================
        # 5️⃣ 双向跨模态注意力
        # =====================================================
        self.power_to_atm_proj = nn.Linear(power_hidden, gnn_hidden)
        self.atm_to_power_proj = nn.Linear(gnn_hidden, power_hidden)

        self.cross_attn_p2a = nn.MultiheadAttention(
            embed_dim=gnn_hidden,
            num_heads=4,
            batch_first=True
        )

        self.cross_attn_a2p = nn.MultiheadAttention(
            embed_dim=power_hidden,
            num_heads=4,
            batch_first=True
        )

        # =====================================================
        # 6️⃣ 融合与预测头
        # =====================================================
        fusion_dim = power_hidden + gnn_hidden

        self.fusion_norm = nn.LayerNorm(fusion_dim)

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, P)
        )

    def forward(self, x_norm, a_flat):
        """
        x_norm: [B, Y, 1]
        a_flat: [B, sum(layer_dims)]
        """

        B = x_norm.size(0)
        # assert sum(self.layer_dims) == a_flat.shape[1]

        # =====================================================
        # 1️⃣ 大气层拆分编码
        # =====================================================
        atm_nodes = []
        idx = 0

        for i, dim in enumerate(self.layer_dims):
            layer_feat = a_flat[:, idx:idx + dim]  # [B, dim]
            atm_nodes.append(self.atm_encoders[i](layer_feat))
            idx += dim

        nodes = torch.stack(atm_nodes, dim=1)  # [B, L, gnn_hidden]

        # 加物理高度位置编码
        nodes = nodes + self.layer_pos

        # 垂直图建模
        nodes = self.vertical_encoder(nodes)  # [B, L, gnn_hidden]

        # =====================================================
        # 2️⃣ 功率时序建模
        # =====================================================
        p_out, _ = self.lstm(x_norm)  # [B, Y, power_hidden]

        # 时序 Attention 聚合
        attn_score = self.temporal_attn(p_out)  # [B, Y, 1]
        attn_weight = torch.softmax(attn_score, dim=1)
        p_feature = torch.sum(attn_weight * p_out, dim=1, keepdim=True)
        # [B, 1, power_hidden]

        # =====================================================
        # 3️⃣ 双向 Cross Attention
        # =====================================================

        # ---- Power → Atmosphere ----
        query_p = self.power_to_atm_proj(p_feature)
        context_p2a, attn_p2a = self.cross_attn_p2a(
            query_p, nodes, nodes
        )
        context_p2a = context_p2a.squeeze(1)  # [B, gnn_hidden]

        # ---- Atmosphere → Power ----
        nodes_proj = self.atm_to_power_proj(nodes)
        context_a2p, attn_a2p = self.cross_attn_a2p(
            p_feature, nodes_proj, nodes_proj
        )
        context_a2p = context_a2p.squeeze(1)  # [B, power_hidden]

        # =====================================================
        # 4️⃣ 融合
        # =====================================================
        combined = torch.cat(
            [context_a2p, context_p2a],
            dim=-1
        )  # [B, fusion_dim]

        combined = self.fusion_norm(combined)

        preds = self.regressor(combined)  # [B, P]

        # return preds.unsqueeze(-1), {
        #     "attn_power_to_atm": attn_p2a,
        #     "attn_atm_to_power": attn_a2p
        # }
        return preds.unsqueeze(-1)

# =========================
# 3️⃣ 参数（你只改这里）
# =========================
MODEL_PATH = "VAPpth/AWPL_checkpoints_300_900_30_Fusion_W2.pth"   # 🔴改成你的pth
ONNX_PATH = "AWPL_checkpoints_300_900_30_Fusion_W2_model.onnx"

Yhis = 300
Phis = 900
rate = 30

Y = Yhis // rate
P = Phis // rate

ERA_FILES = [
    "era5_pressure_merged_1hPa_filtered_normalized",
    "era5_pressure_merged_5hPa_filtered_normalized",
    "era5_pressure_merged_30hPa_filtered_normalized",
    "era5_pressure_merged_200hPa_filtered_normalized",
    "era5_pressure_merged_650hPa_filtered_normalized",
    "era5_surface_merged_all_filter_normalized",
]

SPLIT_DIR = "./splits"

layer_dims = get_layer_dims(SPLIT_DIR, ERA_FILES)
atm_dim = sum(layer_dims)





sess = ort.InferenceSession("AWPL_checkpoints_300_900_30_Fusion_W2_model.onnx")

x = np.random.randn(1, Y, 1).astype(np.float32)
a = np.random.randn(1, atm_dim).astype(np.float32)

out = sess.run(None, {
    "power_seq": x,
    "atm": a
})

print(out[0].shape)