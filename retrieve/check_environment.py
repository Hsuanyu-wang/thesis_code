#!/usr/bin/env python3
"""
Environment check script for embedding computation
This script helps diagnose xformers and related dependency issues
"""

import sys
import subprocess
import importlib

def check_package(package_name, required_version=None):
    """Check if a package is installed and optionally verify version"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        
        if required_version and version != required_version:
            print(f"  ⚠️  Warning: Expected version {required_version}, got {version}")
            return False
        return True
    except ImportError:
        print(f"✗ {package_name}: Not installed")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA: Available ({torch.cuda.get_device_name(0)})")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("✗ CUDA: Not available")
            return False
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        return False

def test_xformers():
    """Test xformers functionality"""
    try:
        import torch
        import xformers
        
        # Test basic xformers functionality
        batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
        
        # Test memory efficient attention
        output = xformers.ops.memory_efficient_attention(q, k, v)
        print("✓ xformers memory efficient attention: Working")
        return True
    except Exception as e:
        print(f"✗ xformers test failed: {e}")
        return False

def test_gte_model():
    """Test GTE model loading"""
    try:
        from transformers import AutoModel, AutoTokenizer
        
        model_path = 'Alibaba-NLP/gte-large-en-v1.5'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            unpad_inputs=True,
            use_memory_efficient_attention=True
        )
        print("✓ GTE model loading: Successful")
        return True
    except Exception as e:
        print(f"✗ GTE model loading failed: {e}")
        return False

def main():
    """Main environment check"""
    print("🔍 Environment Check for Embedding Computation")
    print("=" * 50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check core packages
    print("\n📦 Core Packages:")
    packages = [
        ('torch', '2.1.0'),
        ('transformers', '4.43.2'),
        ('xformers', '0.0.23'),
        ('accelerate', '0.32.1'),
        ('datasets', '2.20.0'),
        ('numpy', '1.24.2'),
    ]
    
    all_good = True
    for package, version in packages:
        if not check_package(package, version):
            all_good = False
    
    # Check CUDA
    print("\n🖥️  CUDA Support:")
    cuda_available = check_cuda()
    
    # Test xformers if CUDA is available
    if cuda_available:
        print("\n🧪 xformers Test:")
        xformers_working = test_xformers()
    else:
        print("\n⚠️  Skipping xformers test (CUDA not available)")
        xformers_working = False
    
    # Test GTE model
    print("\n🤖 GTE Model Test:")
    gte_working = test_gte_model()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Summary:")
    
    if all_good and cuda_available and xformers_working and gte_working:
        print("✅ All checks passed! You can run: python emb.py -d cwq")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        
        if not all_good:
            print("\n💡 To fix package issues, run:")
            print("bash install_emb_deps.sh")
        
        if not cuda_available:
            print("\n💡 CUDA is required for GPU acceleration")
        
        if not xformers_working:
            print("\n💡 xformers issue detected. Try reinstalling with:")
            print("pip uninstall xformers -y")
            print("pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    main() 