#!/usr/bin/env python3
"""
CUDA and Flash Attention capability check for RTX 4090.
Run this script to verify hardware and PyTorch setup before training.
"""
import torch


def check_cuda() -> None:
    """Check CUDA availability and device capabilities."""
    print("=" * 60)
    print("CUDA & Hardware Check")
    print("=" * 60)

    # Basic CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA Available: {cuda_available}")

    if not cuda_available:
        print("✗ CUDA not available. Please check your PyTorch installation.")
        print("  Install with: uv pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return

    # GPU Details
    device_count = torch.cuda.device_count()
    print(f"✓ GPU Count: {device_count}")

    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_capability = torch.cuda.get_device_capability(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)

        print(f"\nGPU {i}:")
        print(f"  Name: {device_name}")
        print(f"  Compute Capability: {device_capability[0]}.{device_capability[1]}")
        print(f"  Total Memory: {total_memory:.2f} GB")

        # Check for RTX 4090 expected specs
        if "4090" in device_name and total_memory < 23.0:
            print(f"  ⚠ Warning: Expected ~24GB VRAM for RTX 4090, found {total_memory:.2f}GB")

    # PyTorch version
    print(f"\n✓ PyTorch Version: {torch.__version__}")

    # Flash Attention check
    print("\n" + "=" * 60)
    print("Flash Attention Support")
    print("=" * 60)

    try:
        # Check if Flash Attention (SDPA) is enabled
        flash_enabled = torch.backends.cuda.flash_sdp_enabled()
        mem_efficient_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
        math_sdp_enabled = torch.backends.cuda.math_sdp_enabled()

        print(f"✓ Flash Attention (SDPA): {flash_enabled}")
        print(f"✓ Memory-Efficient Attention: {mem_efficient_enabled}")
        print(f"✓ Math SDPA Fallback: {math_sdp_enabled}")

        if not flash_enabled:
            print("\n⚠ Flash Attention is disabled. Performance may be suboptimal.")
            print("  Enable with: torch.backends.cuda.enable_flash_sdp(True)")
        else:
            print("\n✓ Flash Attention is ready for optimal performance!")

    except Exception as e:
        print(f"✗ Error checking Flash Attention: {e}")
        print("  This may indicate an incompatible PyTorch version.")

    # BFloat16 support check
    print("\n" + "=" * 60)
    print("BFloat16 Support")
    print("=" * 60)

    # Test BFloat16 tensor creation
    try:
        test_tensor = torch.randn(2, 2, dtype=torch.bfloat16, device='cuda')
        print(f"✓ BFloat16 tensor creation: SUCCESS")
        print(f"  Dtype: {test_tensor.dtype}")
        del test_tensor
    except Exception as e:
        print(f"✗ BFloat16 not supported: {e}")

    # Recommended settings
    print("\n" + "=" * 60)
    print("Recommended Settings for LeJEPA Training")
    print("=" * 60)
    print("• Precision: BFloat16 (torch.bfloat16)")
    print("• Batch Size: 2048-4096 (for SIGReg stability)")
    print("• Compile Mode: torch.compile(model, mode='reduce-overhead')")
    print("• DataLoader: pin_memory=True, persistent_workers=True")
    print("• Gradient Scaler: Optional with BF16 (unlike FP16)")
    print("=" * 60)


if __name__ == "__main__":
    check_cuda()
