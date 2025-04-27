# hardware_utils.py

import torch
import psutil
import platform
from typing import Optional


class TrainingEnvironment:
    def __init__(self, device, multi_gpu: bool, use_amp: bool, batch_size: int, num_workers: int, use_compile: bool = False):
        self.device = device
        self.multi_gpu = multi_gpu
        self.use_amp = use_amp
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_compile = use_compile
        self.benchmark_mode = False

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def autocast(self):
        if self.use_amp:
            return torch.cuda.amp.autocast()
        else:
            # Dummy context manager if AMP is not used
            from contextlib import nullcontext
            return nullcontext()

    def scaler_step(self, loss, optimizer):
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()


def estimate_batch_size(vram_total_gb):
    """
    Estimate a reasonable batch size based on available VRAM.
    """
    if vram_total_gb >= 40:
        return 256
    elif vram_total_gb >= 24:
        return 128
    elif vram_total_gb >= 16:
        return 64
    elif vram_total_gb >= 12:
        return 32
    elif vram_total_gb >= 8:
        return 16
    else:
        return 8


def setup_training_environment(
    target_batch_size: Optional[int] = None,
    prefer_amp: bool = True,
    prefer_compile: bool = True
) -> TrainingEnvironment:
    """
    Detects hardware and sets the optimal training configuration.

    Args:
        target_batch_size: Batch size to aim for, if None, auto-infer based on VRAM.
        prefer_amp: Whether to prefer automatic mixed precision if available.

    Returns:
        A TrainingEnvironment object with device, AMP, and DataLoader info.
    """
    print("\n=== Hardware Detection ===")

    # CPU and RAM
    cpu_count = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"CPU Cores: {cpu_count}")
    print(f"RAM: {ram_gb:.2f} GB")

    # GPU Detection
    multi_gpu = False
    use_amp = False
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        multi_gpu = gpu_count > 1
        total_vram = sum(torch.cuda.get_device_properties(
            i).total_memory for i in range(gpu_count)) / (1024**3)
        print(f"GPU(s): {gpu_count} detected ({total_vram:.2f} GB VRAM total)")

        # Batch size heuristic based on VRAM
        if target_batch_size is None:
            batch_size = estimate_batch_size(total_vram)
        else:
            batch_size = target_batch_size

        # AMP if available and desired
        if prefer_amp:
            if torch.cuda.is_bf16_supported():
                use_amp = True
                print("Automatic Mixed Precision (bfloat16) supported and enabled.")
            else:
                use_amp = True
                print("Automatic Mixed Precision (float16) supported and enabled.")

    else:
        print("No GPU detected. Using CPU.")
        batch_size = 1  # Conservative

    # DataLoader num_workers heuristic
    if device.type == "cuda":
        num_workers = min(8, cpu_count)
    else:
        num_workers = min(4, cpu_count)

    use_compile = False
    if prefer_compile:
        try:
            _ = torch.compile
            use_compile = True
            print("Torch.compile detected and will be used.")
        except AttributeError:
            print("Torch.compile not available in this PyTorch version.")

    print(f"Selected device: {device}")
    print(f"Multi-GPU enabled: {multi_gpu}")
    print(f"Batch Size: {batch_size}")
    print(f"DataLoader Workers: {num_workers}")
    print("===========================\n")

    return TrainingEnvironment(device, multi_gpu, use_amp, batch_size, num_workers, use_compile)
