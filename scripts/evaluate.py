import sys
import os
# 将父目录加入系统路径，以便导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

from data.dataset import PendulumDataset
from utils.functions import load_model_class

console = Console()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 默认评估路径：可以直接使用生成的训练集，或者你单独生成一个测试集
    eval_data_dir = cfg.generation.save_dir 
    model_path = os.path.join("..", "best_model.pt") # 假设在项目根目录运行或指定绝对路径
    
    # 如果根目录找不到，尝试当前运行目录
    if not os.path.exists(model_path):
        model_path = "best_model.pt"
        
    console.print(Panel.fit(
        "[bold cyan]Model Evaluation & Analysis[/bold cyan]\n"
        f"Evaluating on: {eval_data_dir}\n"
        f"Weights: {model_path}",
        border_style="blue"
    ))

    if not os.path.exists(model_path):
        console.print(f"[bold red]❌ Model weights not found at {model_path}![/]")
        return
        
    if not os.path.exists(eval_data_dir):
        console.print(f"[bold red]❌ Evaluation data not found at {eval_data_dir}![/]")
        return

    # 1. 加载模型
    ModelClass = load_model_class(cfg.model.identifier)
    model = ModelClass(cfg.model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    console.print("[bold green]✅ Model loaded successfully.[/]")

    # 2. 加载数据
    try:
        dataset = PendulumDataset(eval_data_dir)
        dataloader = DataLoader(dataset, batch_size=cfg.train.global_batch_size, shuffle=False, num_workers=4)
    except Exception as e:
        console.print(f"[bold red]Error loading dataset: {e}[/]")
        return

    # 3. 执行评估
    all_preds = []
    all_trues = []
    
    console.print(f"🔍 Starting evaluation on {len(dataset)} samples...")
    
    with torch.no_grad():
        for x, y in track(dataloader, description="Evaluating..."):
            x = x.to(device)
            pred = model(x)
            
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    # 4. 计算指标
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(all_preds - all_trues), axis=0)
    
    # MAPE (Mean Absolute Percentage Error) - 避免除以0
    epsilon = 1e-8
    mape = np.mean(np.abs((all_trues - all_preds) / (all_trues + epsilon)), axis=0) * 100

    # 5. 打印结果表格
    results_table = Table(title="📊 Evaluation Metrics", show_lines=True)
    results_table.add_column("Parameter", justify="center", style="cyan", no_wrap=True)
    results_table.add_column("MAE (Absolute Error)", justify="right", style="bold green")
    results_table.add_column("MAPE (Relative Error %)", justify="right", style="bold yellow")

    results_table.add_row("K1 (Linear Friction)", f"{mae[0]:.6f}", f"{mape[0]:.2f}%")
    results_table.add_row("K2 (Quadratic Drag)", f"{mae[1]:.6f}", f"{mape[1]:.2f}%")
    
    console.print("\n")
    console.print(results_table)

    # 6. 绘制预测散点图
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制 K1
    ax1.scatter(all_trues[:, 0], all_preds[:, 0], alpha=0.5, color='#00f3ff', s=10)
    # 添加 y=x 参考线
    min_k1, max_k1 = np.min(all_trues[:, 0]), np.max(all_trues[:, 0])
    ax1.plot([min_k1, max_k1], [min_k1, max_k1], 'r--', lw=2, label="Ideal (y=x)")
    ax1.set_title('K1: Prediction vs Ground Truth', fontsize=14, color='white')
    ax1.set_xlabel('Ground Truth K1', fontsize=12)
    ax1.set_ylabel('Predicted K1', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # 绘制 K2
    ax2.scatter(all_trues[:, 1], all_preds[:, 1], alpha=0.5, color='#0aff60', s=10)
    # 添加 y=x 参考线
    min_k2, max_k2 = np.min(all_trues[:, 1]), np.max(all_trues[:, 1])
    ax2.plot([min_k2, max_k2], [min_k2, max_k2], 'r--', lw=2, label="Ideal (y=x)")
    ax2.set_title('K2: Prediction vs Ground Truth', fontsize=14, color='white')
    ax2.set_xlabel('Ground Truth K2', fontsize=12)
    ax2.set_ylabel('Predicted K2', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = "evaluation_plot.png"
    plt.savefig(save_path, dpi=300)
    console.print(f"\n[bold green]📈 Scatter plot saved to {save_path}[/]")

if __name__ == "__main__":
    main()