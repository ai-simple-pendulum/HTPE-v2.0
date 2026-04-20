import os
import sys
import torch
import torch.nn as nn

# 🚨 【核心修复】将项目根目录 (HTPE) 加入路径，这样才能识别到 'models' 包
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 确保权重文件路径正确（权重在根目录下，所以要回退一级）
WEIGHT_PATH = os.path.join(project_root, "best_model.pt")

# 现在导入就不会报错了
try:
    from models.transformer import ModernPendulumTransformer
except ImportError:
    # 兼容性备选方案
    from transformer import ModernPendulumTransformer

# ==========================================
# 1. 构造配置类 (根据你的代码需求)
# ==========================================
class DummyConfig:
    def __init__(self):
        # ⚠️ 请确认这些参数与你训练时一致！
        self.model_dim = 256    
        self.num_heads = 8      
        self.dropout = 0.0      
        self.num_layers = 4     
        self.output_dim = 3     

def export_htpe_to_onnx():
    print("⏳ 正在根据权重维度初始化模型架构...")
    cfg = DummyConfig()
    model = ModernPendulumTransformer(cfg)
    
    print(f"⏳ 正在加载权重: {WEIGHT_PATH}")
    try:
        checkpoint = torch.load(WEIGHT_PATH, map_location="cpu")
        
        # 自动处理各种保存格式 (有些保存的是整个 dict, 有些只是 state_dict)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 自动处理 DDP 前缀 'module.'
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict)
        print("✅ 权重加载成功！")
    except Exception as e:
        print(f"❌ 加载失败，详细错误:\n{e}")
        return

    model.eval()

    # 构造伪输入 (必须符合 CNN Backbone 降采样逻辑)
    # 输入窗口建议给长一点，比如 2048
    SEQ_LEN = 2048 
    dummy_input = torch.randn(1, SEQ_LEN, 1, dtype=torch.float32)

    onnx_file_path = os.path.join(project_root, "htpe_model.onnx")
    print(f"🚀 开始导出 ONNX -> {onnx_file_path}")
    
    # 注意：导出时保持 opset_version=14 以支持 SDPA 算子
    torch.onnx.export(
        model,                      
        dummy_input,                
        onnx_file_path,             
        export_params=True,         
        opset_version=14,           
        do_constant_folding=True,   
        input_names=['input_theta'],
        output_names=['out_params'], 
    )
    print(f"🎉 导出成功！文件位置: {onnx_file_path}")

if __name__ == "__main__":
    export_htpe_to_onnx()