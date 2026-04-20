import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "facebook/opt-125m"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = "cuda"
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    inputs = tokenizer("Test prompt", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    print("forward pass ok")
    print("logits shape:", tuple(outputs.logits.shape))


if __name__ == "__main__":
    main()
