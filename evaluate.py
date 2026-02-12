import os
import pickle
import random
import textwrap
import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from train import Vocabulary, ImageCaptioner, greedy_search, load_captions, compute_bleu4


def load_model(checkpoint_path="caption_model.pt", device="cpu"):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train.py first.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]
    model = ImageCaptioner(len(vocab.word2idx)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    history = checkpoint.get("history", {})
    return model, vocab, history


def compute_precision_recall_f1(references, hypotheses):
    precision_list, recall_list, f1_list = [], [], []
    for reference, hypothesis in zip(references, hypotheses):
        reference_words = set(reference.split())
        hypothesis_words = set(hypothesis.split())
        if not hypothesis_words:
            precision_list.append(0.0)
            recall_list.append(0.0)
            f1_list.append(0.0)
            continue
        true_positives = len(reference_words & hypothesis_words)
        precision = true_positives / len(hypothesis_words)
        recall = true_positives / len(reference_words) if reference_words else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)
    return sum(precision_list) / len(precision_list), sum(recall_list) / len(recall_list), sum(f1_list) / len(f1_list)


def caption_examples(model, vocab, features, pairs, images_dir, num_examples=5, device="cpu"):
    indices = list(range(len(pairs)))
    random.shuffle(indices)
    examples = []
    for index in indices[:num_examples]:
        image_name, ground_truth = pairs[index]
        if image_name not in features:
            continue
        predicted_caption = greedy_search(model, torch.tensor(features[image_name]), vocab, device=device)
        image_path = os.path.join(images_dir, image_name) if images_dir else None
        if image_path and not os.path.isfile(image_path):
            image_path = None
        examples.append({"image": image_name, "image_path": image_path, "ground_truth": ground_truth, "predicted": predicted_caption})
    return examples


def plot_loss_curve(history, out_path=None):
    if not history.get("train_loss") and not history.get("val_loss"):
        print("No loss history. Run train.py first.")
        return
    output_path = out_path or "loss_curve.png"
    plt.figure(figsize=(8, 5))
    if history.get("train_loss"):
        plt.plot(history["train_loss"], label="Train Loss")
    if history.get("val_loss"):
        plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    print(f"Loss curve saved to {output_path}")


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, history = load_model(device=str(device))
    pairs = load_captions("data/captions.txt")
    with open("flickr30k_features.pkl", "rb") as features_file:
        features = pickle.load(features_file)
    pairs = [(image_name, caption) for image_name, caption in pairs if image_name in features]

    print("\n--- Caption Examples ---")
    examples = caption_examples(model, vocab, features, pairs, "data/Images", num_examples=5, device=device)
    for example in examples:
        print(f"\nImage: {example['image']}")
        print(f"  Ground Truth: {example['ground_truth']}")
        print(f"  Predicted:    {example['predicted']}")

    print("\n--- Loss Curve ---")
    plot_loss_curve(history)

    print("\n--- Metrics ---")
    sample_indices = random.sample(range(len(pairs)), min(500, len(pairs)))
    references = [pairs[index][1] for index in sample_indices]
    hypotheses = [greedy_search(model, torch.tensor(features[pairs[index][0]]), vocab, device=device) for index in tqdm(sample_indices, desc="Evaluating")]
    bleu_score = compute_bleu4(references, hypotheses)
    precision, recall, f1_score = compute_precision_recall_f1(references, hypotheses)
    print(f"BLEU-4:    {bleu_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1_score:.4f}")

    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for index, example in enumerate(examples[:5]):
        axis = axes[index]
        if example["image_path"]:
            image = Image.open(example["image_path"]).convert("RGB")
            axis.imshow(image)
        axis.set_title(example["image"][:20] + "...")
        axis.axis("off")
        gt_wrapped = textwrap.fill(example["ground_truth"], width=50)
        pred_wrapped = textwrap.fill(example["predicted"], width=50)
        axis.text(0.5, -0.15, f"GT: {gt_wrapped}\nPred: {pred_wrapped}", transform=axis.transAxes, fontsize=8, ha="center", va="top")
    axes[5].axis("off")
    plt.suptitle("Caption Examples")
    plt.tight_layout()
    plt.savefig("caption_examples.png", bbox_inches="tight")
    print("\nFigure saved to caption_examples.png")


if __name__ == "__main__":
    run_evaluation()
