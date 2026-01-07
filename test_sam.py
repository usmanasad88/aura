import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def test_setup():
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        # This will attempt to load the model (requires HF login)
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        print("✅ SAM 3 successfully initialized!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        print("Hint: Ensure you have run 'huggingface-cli login' and have access to the repo.")

if __name__ == "__main__":
    test_setup()