import torch
import torch.nn as nn
from models.encoders import img_tokenizer
import time

def test_dinov2_patch_embeddings():
    """Test DINOv2 patch embeddings extraction"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy input
    batch_size = 2
    channels = 3
    height, width = 224, 224
    
    dummy_img = torch.randn(batch_size, channels, height, width).to(device)
    print(f"Input image shape: {dummy_img.shape}")
    
    # Initialize the img_tokenizer
    tokenizer = img_tokenizer(
        img_size=224,
        patch_size=14,  # DINOv2 ViT-S/14 uses 14x14 patches
        in_chans=3,
        embed_dim=384  # DINOv2 ViT-S has 384 embedding dimension
    ).to(device)
    
    print("Model loaded successfully!")
    
    # Test forward pass
    print("\nTesting forward pass...")
    start_time = time.time()
    
    with torch.no_grad():
        patch_embeddings, H, W = tokenizer(dummy_img)
    
    end_time = time.time()
    
    # Print results
    print(f"Forward pass completed in {end_time - start_time:.4f} seconds")
    print(f"Original image size: {H} x {W}")
    print(f"Patch embeddings shape: {patch_embeddings.shape}")
    
    # Calculate expected number of patches
    patch_size = 14
    expected_patches = (H // patch_size) * (W // patch_size)
    print(f"Expected number of patches: {expected_patches}")
    print(f"Actual number of patches: {patch_embeddings.shape[1]}")
    
    # Verify patch embeddings
    if patch_embeddings.shape[1] == expected_patches:
        print("✅ Patch embedding extraction successful!")
    else:
        print("❌ Mismatch in patch count")
    
    # Print some statistics
    print(f"\nPatch embeddings statistics:")
    print(f"Mean: {patch_embeddings.mean().item():.6f}")
    print(f"Std: {patch_embeddings.std().item():.6f}")
    print(f"Min: {patch_embeddings.min().item():.6f}")
    print(f"Max: {patch_embeddings.max().item():.6f}")
    
    return patch_embeddings

def test_different_input_sizes():
    """Test with different input sizes"""
    print("\n" + "="*50)
    print("Testing different input sizes...")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different image sizes
    test_sizes = [(224, 224), (448, 448), (336, 336)]
    
    for img_size in test_sizes:
        print(f"\nTesting with image size: {img_size}")
        
        # Create tokenizer for this size
        tokenizer = img_tokenizer(
            img_size=img_size,
            patch_size=14,
            in_chans=3,
            embed_dim=384
        ).to(device)
        
        # Create dummy input
        dummy_img = torch.randn(1, 3, img_size[0], img_size[1]).to(device)
        
        with torch.no_grad():
            patch_embeddings, H, W = tokenizer(dummy_img)
        
        print(f"Input: {dummy_img.shape}")
        print(f"Output patches: {patch_embeddings.shape}")
        print(f"Expected patches: {(H//14) * (W//14)}")

def test_batch_processing():
    """Test batch processing"""
    print("\n" + "="*50)
    print("Testing batch processing...")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = img_tokenizer(
        img_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=384
    ).to(device)
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8]
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        dummy_img = torch.randn(batch_size, 3, 224, 224).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            patch_embeddings, H, W = tokenizer(dummy_img)
        end_time = time.time()
        
        print(f"Input shape: {dummy_img.shape}")
        print(f"Output shape: {patch_embeddings.shape}")
        print(f"Time taken: {end_time - start_time:.4f}s")
        print(f"Time per sample: {(end_time - start_time)/batch_size:.4f}s")

if __name__ == "__main__":
    print("DINOv2 Patch Embeddings Test")
    print("="*50)
    
    try:
        # Main test
        patch_embeddings = test_dinov2_patch_embeddings()
        
        # Additional tests
        test_different_input_sizes()
        test_batch_processing()
        
        print("\n" + "="*50)
        print("All tests completed successfully! ✅")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
