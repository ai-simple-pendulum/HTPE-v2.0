# -*- coding: utf-8 -*-
import os
import multiprocessing

# ==========================================
# Optimization 1: Threads and Parallelism
# ==========================================
TOTAL_CORES = multiprocessing.cpu_count()
COMPUTE_CORES = max(1, TOTAL_CORES - 6)

os.environ["OMP_NUM_THREADS"] = str(COMPUTE_CORES)
os.environ["MKL_NUM_THREADS"] = str(COMPUTE_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(COMPUTE_CORES)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(COMPUTE_CORES)
os.environ["NUMEXPR_NUM_THREADS"] = str(COMPUTE_CORES)

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, TextColumn, BarColumn, TaskProgressColumn, 
    TimeRemainingColumn, SpinnerColumn, track
)

from data.dataset import PendulumDataset
from utils.functions import load_model_class
from utils.common import set_seed, count_parameters

console = Console()

# ==========================================
# Optimization 2: RAM Caching Wrapper
# ==========================================
class CachedDataset(Dataset):
    def __init__(self, original_dataset, desc="Caching Data"):
        self.data = []
        for i in track(range(len(original_dataset)), description=f"[cyan]{desc}..."):
            self.data.append(original_dataset[i])
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def visualize_config(cfg: DictConfig, compute_cores: int):
    table = Table(title="Experiment Configuration (CPU EXTREME Mode)", show_header=True, header_style="bold magenta")
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Parameter", style="green")
    table.add_column("Value", style="bold white")

    conf_dict = OmegaConf.to_container(cfg, resolve=True)
    
    table.add_row("System", "CPU Cores Allocated", f"[bold red]{compute_cores} / {TOTAL_CORES}[/]")
    table.add_row("System", "Data Loading", "[bold red]RAM Cached (In-Memory)[/]")
    table.add_section()

    for section, params in conf_dict.items():
        if isinstance(params, dict):
            first = True
            for k, v in params.items():
                if isinstance(v, list): v = str(v)
                table.add_row(section if first else "", k, str(v))
                first = False
            table.add_section()
            
    console.print(table)

def visualize_metrics(epoch, train_loss, val_loss, k1_err, k2_err, lr):
    table = Table(box=None, show_header=True)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    
    loss_color = "green" if val_loss < 0.001 else "yellow"
    
    table.add_row("Train Loss", f"{train_loss:.6f}")
    table.add_row("Val Loss", f"[{loss_color}]{val_loss:.6f}[/]")
    table.add_row("K1 Error (Friction)", f"{k1_err:.6f}")
    table.add_row("K2 Error (Drag)", f"{k2_err:.6f}")
    table.add_row("Learning Rate", f"{lr:.2e}")
    
    panel = Panel(table, title=f"Epoch {epoch} Summary", border_style="blue", expand=False)
    console.print(panel)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    torch.set_num_threads(COMPUTE_CORES)
    
    console.print(Panel.fit(
        "[bold cyan]Hybrid Physics-AI Engine[/bold cyan]\n"
        "[dim]Temporal Folding & Resonant Encoding Architecture[/dim]\n"
        f"[bold red]HIGH-PERFORMANCE CPU MODE: {COMPUTE_CORES} CORES[/bold red]",
        border_style="red"
    ))
    
    set_seed(42)
    visualize_config(cfg, COMPUTE_CORES)
    
    console.rule("[bold yellow]Data Loading & RAM Caching[/bold yellow]")
    data_dir = cfg.generation.save_dir
    
    try:
        # 使用你刚刚清理后的干净数据 (36000长度)
        disk_ds = PendulumDataset(data_dir)
        val_size = int(len(disk_ds) * cfg.train.val_split)
        train_disk_ds, val_disk_ds = random_split(disk_ds, [len(disk_ds)-val_size, val_size])
        
        console.print("[yellow]Loading all CSVs into RAM to bypass disk I/O bottleneck...[/]")
        train_ds = CachedDataset(train_disk_ds, desc="Caching Train Set")
        val_ds = CachedDataset(val_disk_ds, desc="Caching Val Set")
        
        console.print(f"Data fully loaded in memory: [bold]{len(disk_ds)}[/] total samples")

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return

    train_loader = DataLoader(train_ds, batch_size=cfg.train.global_batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.global_batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    console.rule("[bold yellow]Model Initialization[/bold yellow]")
    ModelClass = load_model_class(cfg.model.identifier)
    model = ModelClass(cfg.model)
    
    device = torch.device("cpu")
    model.to(device)
    
    # === 已安全移除容易导致崩溃的 torch.compile 代码块 ===
    
    param_count = count_parameters(model)
    console.print(f"Model Architecture: [bold cyan]{cfg.model.identifier}[/]")
    console.print(f"Trainable Parameters: [bold green]{param_count:,}[/]")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        betas=(cfg.train.optimizer.beta1, cfg.train.optimizer.beta2),
        weight_decay=cfg.train.optimizer.weight_decay
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.train.optimizer.lr * 10,
        total_steps=cfg.train.epochs * len(train_loader), pct_start=0.1
    )
    
    criterion = nn.MSELoss()
    
    console.rule("[bold yellow]Training Start (90-Core CPU Mode)[/bold yellow]")
    best_loss = float('inf')
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("[bold blue]{task.fields[info]}", justify="right"),
    )

    use_amp = True
    
    with progress:
        epoch_task = progress.add_task("[green]Total Progress", total=cfg.train.epochs, info="Starting...")
        
        for epoch in range(cfg.train.epochs):
            model.train()
            total_loss = 0
            
            batch_task = progress.add_task(f"Epoch {epoch+1}", total=len(train_loader), info="Loss: 0.000")
            
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_amp):
                    pred = model(x)
                    loss = criterion(pred, y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress.update(batch_task, advance=1, info=f"Loss: {loss.item():.5f}")
            
            progress.remove_task(batch_task)
            avg_train_loss = total_loss / len(train_loader)
            
            if epoch % cfg.train.eval_interval == 0 or epoch == cfg.train.epochs - 1:
                model.eval()
                err_k1, err_k2 = 0, 0
                val_loss_accum = 0
                
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_amp):
                            pred = model(x)
                            val_loss_accum += criterion(pred, y).item()
                        
                        err_k1 += nn.functional.mse_loss(pred[:,0], y[:,0]).item()
                        err_k2 += nn.functional.mse_loss(pred[:,1], y[:,1]).item()
                
                steps = len(val_loader)
                avg_val_loss = val_loss_accum / steps
                avg_k1 = err_k1 / steps
                avg_k2 = err_k2 / steps
                
                progress.stop() 
                visualize_metrics(epoch+1, avg_train_loss, avg_val_loss, avg_k1, avg_k2, optimizer.param_groups[0]['lr'])
                progress.start()

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(model.state_dict(), "best_model_cpu.pt")
                    console.print(f"   [bold green]New Best Model Saved (CPU)! (Loss: {best_loss:.6f})[/]")

            progress.update(epoch_task, advance=1, info=f"Best: {best_loss:.5f}")

    console.rule("[bold green]Training Completed[/bold green]")
    console.print(f"Best Validation Loss: [bold green]{best_loss:.6f}[/]")

if __name__ == "__main__":
    main()