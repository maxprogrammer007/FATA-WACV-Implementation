import torch
import torch.nn.functional as F
from .augmenter import FATA_Augmenter

def shannon_entropy(p: torch.Tensor) -> torch.Tensor:
    """Calculates entropy of a softmax output tensor p."""
    return -(p * torch.log(p + 1e-8)).sum(1)

def adapt_on_image(image_tensor: torch.Tensor, model, model_part1, model_part2, optimizer, augmenter: FATA_Augmenter):
    """
    Performs one step of Test-Time Adaptation on a single image.
    """
    # The entropy threshold recommended by the EATA/DeYO papers
    entropy_threshold = 0.4 * torch.log(torch.tensor(1000.)) # 1000 classes for ImageNet

    # --- Step 1: Original Forward Pass (to get pseudo-label and check entropy) ---
    z = model_part1(image_tensor)
    outputs_original = model_part2(z)
    softmax_original = F.softmax(outputs_original, dim=1)
    
    # Store this prediction to show the "before" state
    pre_adapt_softmax = softmax_original.detach().clone()

    entropy = shannon_entropy(softmax_original)

    # --- Step 2: Sample Selection (only adapt on confident samples) ---
    if entropy.item() >= entropy_threshold:
        print(f"Skipping adaptation: Entropy {entropy.item():.2f} >= Threshold {entropy_threshold:.2f}")
        # Return 0 loss and the same prediction for "before" and "after"
        return 0.0, pre_adapt_softmax, pre_adapt_softmax

    # --- Step 3: Augmented Forward Pass ---
    z_prime = augmenter.augment(z)
    outputs_augmented = model_part2(z_prime)

    # --- Step 4: Calculate Total Loss ---
    # TTA Loss: Minimize entropy of the original prediction
    loss_tta = entropy.mean()
    
    # FATA Loss: Cross-entropy between augmented output and pseudo-label
    pseudo_labels = torch.argmax(softmax_original, dim=1).detach()
    loss_fata = F.cross_entropy(outputs_augmented, pseudo_labels)
    
    total_loss = loss_tta + loss_fata
    print(f"Adaptation step: Total Loss = {total_loss.item():.4f}")

    # --- Step 5: Optimizer Step ---
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # --- Step 6: Post-Adaptation Forward Pass (to see the result) ---
    with torch.no_grad():
        # Use the full model now that its parameters have been updated
        post_adapt_outputs = model(image_tensor)
        post_adapt_softmax = F.softmax(post_adapt_outputs, dim=1)

    return total_loss.item(), pre_adapt_softmax, post_adapt_softmax

