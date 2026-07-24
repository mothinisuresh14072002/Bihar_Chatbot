"""
Download the Qwen2.5-3B-Instruct GGUF model for the Bihar BOCW RAG Chatbot.
This is a text-only model — faster than vision models, with excellent Hindi/English support.
"""
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import MODEL_PATH, MODEL_FILENAME, MODELS_DIR


def download_model():
    """Download the GGUF model using huggingface_hub."""
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"✅ Model already exists: {MODEL_PATH}")
        print(f"   Size: {size_mb:.1f} MB")
        return

    print("=" * 60)
    print("  BOCW Chatbot — Model Download")
    print("=" * 60)
    print(f"  Model : Qwen2.5-3B-Instruct (Q4_K_M quantization)")
    print(f"  Size  : ~2.0 GB")
    print(f"  Target: {MODEL_PATH}")
    print("=" * 60)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    print("\n⬇️  Downloading model... (this may take a few minutes)\n")

    try:
        downloaded_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
            filename=MODEL_FILENAME,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"\n✅ Model downloaded successfully!")
        print(f"   Path: {downloaded_path}")

        # Verify
        actual = Path(downloaded_path)
        if actual.exists():
            size_mb = actual.stat().st_size / (1024 * 1024)
            print(f"   Size: {size_mb:.1f} MB")
        else:
            print("⚠️  Downloaded but file not found at expected path.")
            print(f"   Check: {MODELS_DIR}")

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\n📋 Manual download instructions:")
        print(f"   1. Go to: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF")
        print(f"   2. Download: {MODEL_FILENAME}")
        print(f"   3. Place it in: {MODELS_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
