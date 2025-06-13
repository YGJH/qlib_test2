from pathlib import Path
import torch
from qlib.contrib.model.pytorch_hist import HIST

# 一定要给全量跟 main.py 一致的参数，
# 包括 stock2concept、stock_index、n_epochs、metric、early_stop、loss、GPU
HOME = Path.home() / ".qlib" / "qlib_data" / "my_us_data"
work_dir = Path(__file__).parent / "snapshot_hist_clean_v8"
work_dir.mkdir(parents=True, exist_ok=True)

model = HIST(
    d_feat=360,
    hidden_size=128,
    num_layers=3,
    dropout=0.1,
    n_epochs=100,
    lr=0.0005,
    metric="ic",
    early_stop=15,
    loss="mse",
    base_model="GRU",
    stock2concept=str(HOME/"stock2concept.npy"),
    stock_index=str(HOME/"stock_index.npy"),
    optimizer="adam",
    GPU=0,
    seed=42,
    work_dir=str(work_dir),
)

ckpt_path = work_dir / "hist_us.pth"
torch.save(model.HIST_model.state_dict(), ckpt_path)
print(f"✓ 已生成符合 main.py 配置的 checkpoint: {ckpt_path}")