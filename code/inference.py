import os
import random

import torch
from PIL import Image
from transformers import CLIPProcessor, AutoTokenizer

from config import DEVICE, CHECKPOINT_DIR, VISION_MODEL_NAME, LANGUAGE_MODEL_NAME, IMAGE_SIZE
from model import MiniBLIP2


def load_model(checkpoint_path=None):
    """Load trained Mini-BLIP2 model from checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    model = MiniBLIP2().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (loss={checkpoint['loss']:.4f})")
    return model


def generate_caption(model, image_path, clip_processor, tokenizer,
                     use_beam: bool = False, num_beams: int = 3):
    """Generate caption for a single image."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    if use_beam:
        return model.generate_beam(pixel_values, tokenizer, num_beams=num_beams)
    return model.generate(pixel_values, tokenizer)


def run_inference(image_dir, captions_dict, num_samples=5, use_beam=False, num_beams=3,
                  process_all=False):
    """Generate captions for images and print results."""
    model = load_model()
    clip_processor = CLIPProcessor.from_pretrained(VISION_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not all_images:
        print(f"No images found in {image_dir}")
        return

    if process_all:
        samples = sorted(all_images)
    else:
        samples = random.sample(all_images, min(num_samples, len(all_images)))

    method = f"Beam Search (k={num_beams})" if use_beam else "Greedy"
    print(f"\n{'='*80}")
    print(f"Generation Results — {method}")
    print(f"{'='*80}")

    for i, img_name in enumerate(samples):
        img_path = os.path.join(image_dir, img_name)
        gen_caption = generate_caption(model, img_path, clip_processor, tokenizer,
                                       use_beam=use_beam, num_beams=num_beams)

        # Get ground truth caption if available
        true_caption = None
        if captions_dict and img_name in captions_dict:
            true_caption = captions_dict[img_name][0]  # first caption

        print(f"\n[{i+1}] Image: {img_name}")
        if true_caption:
            print(f"    True: {true_caption}")
        print(f"    Gen:  {gen_caption}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    import argparse
    from data_loader import load_captions, IMAGE_DIR, CAPTIONS_FILE

    parser = argparse.ArgumentParser(description="Mini-BLIP2 Image Captioning")
    parser.add_argument("--beam", type=int, default=0,
                        help="Beam size for beam search (0 = greedy, >0 = beam search)")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of images to sample (ignored when --image_dir is set)")
    import_images_dir = os.path.join(os.path.dirname(IMAGE_DIR), "import_images")
    parser.add_argument("--image_dir", type=str, default=import_images_dir,
                        help=f"Image directory (default: {import_images_dir})")
    args = parser.parse_args()

    image_dir = args.image_dir
    image_captions = None
    process_all = True

    use_beam = args.beam > 0
    num_beams = args.beam if use_beam else 3
    run_inference(image_dir, image_captions, num_samples=args.samples,
                  use_beam=use_beam, num_beams=num_beams, process_all=process_all)
