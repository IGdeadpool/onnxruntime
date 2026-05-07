import os
from pathlib import Path

from datasets import load_dataset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path("/home/l/benchmarks")
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

CIFAR10_DIR = DATA_DIR / "cifar10"
HF_DATASETS_DIR = DATA_DIR / "huggingface_datasets"
TORCH_MODEL_DIR = MODEL_DIR / "torch"
HF_MODEL_DIR = MODEL_DIR / "huggingface"

DISTILBERT_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

# Hugging Face is often unreachable from WSL networks in China. The official
# libraries honor HF_ENDPOINT, so default to the public mirror unless the user
# explicitly provides another endpoint.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def ensure_dirs() -> None:
    for path in [
        CIFAR10_DIR,
        HF_DATASETS_DIR,
        TORCH_MODEL_DIR,
        HF_MODEL_DIR,
        ROOT / "scripts",
        ROOT / "outputs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def download_resnet18() -> None:
    print("[1/4] Downloading ResNet18 ImageNet weights...")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).eval()
    print(f"      ResNet18 ready: {weights}")
    print(f"      Torch cache: {TORCH_MODEL_DIR}")
    del model


def download_cifar10() -> None:
    print("[2/4] Downloading CIFAR-10 train/test...")
    weights = ResNet18_Weights.DEFAULT
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=weights.transforms().mean,
                std=weights.transforms().std,
            ),
        ]
    )

    train = datasets.CIFAR10(
        root=str(CIFAR10_DIR),
        train=True,
        download=True,
        transform=transform,
    )
    test = datasets.CIFAR10(
        root=str(CIFAR10_DIR),
        train=False,
        download=True,
        transform=transform,
    )
    print(f"      CIFAR-10 ready: train={len(train)} test={len(test)}")
    print(f"      CIFAR-10 path: {CIFAR10_DIR}")


def download_sst2() -> None:
    print("[3/4] Downloading GLUE/SST-2 dataset...")
    sst2 = load_dataset(
        "glue",
        "sst2",
        cache_dir=str(HF_DATASETS_DIR),
    )
    print(f"      SST-2 ready: {dict((k, len(v)) for k, v in sst2.items())}")
    print(f"      SST-2 cache: {HF_DATASETS_DIR}")


def download_distilbert() -> None:
    print("[4/4] Downloading DistilBERT SST-2 model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        DISTILBERT_MODEL_ID,
        cache_dir=str(HF_MODEL_DIR),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_ID,
        cache_dir=str(HF_MODEL_DIR),
    ).eval()
    print(f"      DistilBERT ready: {DISTILBERT_MODEL_ID}")
    print(f"      Vocab size: {tokenizer.vocab_size}")
    print(f"      Hugging Face model cache: {HF_MODEL_DIR}")
    del model


def print_summary() -> None:
    print("\nDownload targets:")
    print(f"  ResNet18 weights: {TORCH_MODEL_DIR}/hub/checkpoints")
    print(f"  CIFAR-10:         {CIFAR10_DIR}")
    print(f"  SST-2:            {HF_DATASETS_DIR}")
    print(f"  DistilBERT:       {HF_MODEL_DIR}")

    print("\nRecommended shell exports before running:")
    print(f"  export TORCH_HOME={TORCH_MODEL_DIR}")
    print(f"  export HF_HOME={HF_MODEL_DIR}")
    print(f"  export TRANSFORMERS_CACHE={HF_MODEL_DIR}/transformers")
    print(f"  export HF_DATASETS_CACHE={HF_DATASETS_DIR}")
    print("  export HF_ENDPOINT=https://hf-mirror.com")
    print("  export HF_HUB_DISABLE_XET=1")


def main() -> None:
    ensure_dirs()
    print_summary()
    print()
    download_resnet18()
    download_cifar10()
    download_sst2()
    download_distilbert()
    print("\nAll baseline assets are ready.")


if __name__ == "__main__":
    main()
