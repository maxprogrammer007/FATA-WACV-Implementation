import torch
import torchvision.transforms as T
from PIL import Image
import requests

from fata.model import setup_model
from fata.augmenter import FATA_Augmenter
from fata.adapt import adapt_on_image
from utils.corruptions import add_gaussian_noise

def main():
    """
    A simple command-line interface to test the FATA adaptation process.
    """
    print("--- FATA CLI Test Script ---")

    # 1. Setup model and augmenter
    model, model_part1, model_part2, optimizer = setup_model()
    augmenter = FATA_Augmenter()

    # 2. Load a sample image
    url = "https://images.unsplash.com/photo-1583337130417-3346a1be7dee?auto=format&fit=crop&w=500"
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        print("Sample image loaded.")
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    # 3. Corrupt the image
    corrupted_image = add_gaussian_noise(image, level=50)
    print("Image corrupted with noise.")

    # 4. Prepare image tensor for the model
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(corrupted_image).unsqueeze(0) # Add batch dimension

    # 5. Run one step of adaptation
    print("\nRunning adaptation...")
    loss, pre_pred, post_pred = adapt_on_image(
        image_tensor, model, model_part1, model_part2, optimizer, augmenter
    )

    # 6. Show results
    print(f"\nAdaptation finished with loss: {loss:.4f}")
    pre_conf, pre_class = torch.max(pre_pred, 1)
    post_conf, post_class = torch.max(post_pred, 1)

    print(f"Prediction BEFORE adaptation: Class {pre_class.item()} with confidence {pre_conf.item():.2%}")
    print(f"Prediction AFTER adaptation:  Class {post_class.item()} with confidence {post_conf.item():.2%}")

if __name__ == "__main__":
    main()
