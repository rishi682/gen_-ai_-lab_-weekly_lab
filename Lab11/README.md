# GPT-2 Fine-Tuning Lab

Fine-tune a pre-trained GPT-2 model for two real-world industry applications: a **Product Review Generator** for e-commerce and a **Recipe Instruction Generator** for food-tech. The lab demonstrates how transfer learning adapts a general language model to a specific business domain using only a small dataset.

---

## What You Will Learn

- How fine-tuning applies to real-world industry applications
- How to load and configure GPT-2 using Hugging Face Transformers
- How to prepare domain-specific datasets for causal language modeling
- How to compare model output before and after fine-tuning
- How to evaluate generation quality using perplexity

---

## Lab Structure

| Component | Domain | Task |
|-----------|--------|------|
| Component I | E-Commerce | Product Review Generator |
| Component II | Food-Tech | Recipe Instruction Generator |

Each component follows the same four-step workflow independently:

1. Load model and generate baseline output
2. Prepare and tokenize the domain dataset
3. Fine-tune with the Hugging Face `Trainer`
4. Generate output and compare against baseline

---

## Requirements

All dependencies are installed in the first cell of the notebook:

```
transformers
datasets
accelerate
```

No local installation is needed when running on Colab.

---

## Datasets

Both datasets are embedded directly in the notebook — no external downloads required.

**Component I — Product Reviews (20 sentences)**
Short consumer review sentences covering electronics and gadgets. Typical vocabulary: `battery life`, `value for money`, `highly recommend`, `build quality`.

**Component II — Recipe Instructions (20 sentences, 4 dishes)**
Step-by-step cooking instructions for butter chicken, pasta carbonara, vegetable stir fry, and chocolate chip cookies. Typical vocabulary: `marinate`, `cook on medium heat`, `serve hot`, `bake at 180 degrees`.

---

## Training Configuration

Both components use identical hyperparameters:

| Parameter | Value |
|-----------|-------|
| Base model | `gpt2` (117M parameters) |
| Epochs | 15 |
| Learning rate | 5e-5 |
| Batch size | 4 |
| Max sequence length | 128 tokens |
| Weight decay | 0.01 |
| Warmup steps | 50 |
| Mixed precision | fp16 (when GPU available) |

---

## Expected Output

**Component I — Product Reviews**

Before fine-tuning, prompts like `"This product is"` produce generic news or Wikipedia-style text. After fine-tuning, the model generates e-commerce review language:

```
This product is excellent value for money and the build quality feels very solid.
I bought this phone and the camera quality is outstanding for the price point.
```

**Component II — Recipe Instructions**

Before fine-tuning, prompts like `"To make butter chicken"` produce unrelated text. After fine-tuning, the model generates sequential cooking steps:

```
To make butter chicken start by marinating the chicken pieces in yogurt and spices for one hour.
For pasta carbonara boil spaghetti until al dente then toss with pancetta and the egg mixture.
```

---

## Evaluation

Perplexity is computed after training using the held-out test split:

```
Perplexity = exp(eval_loss)
```

A lower perplexity score means the model assigns higher probability to the domain text — it is less "surprised" by it. Expect a significant drop compared to the untrained baseline.

---

## Alternative Models

You can replace `'gpt2'` with any of the following lightweight models for faster training or stronger output:

| Model ID | Size | Notes |
|----------|------|-------|
| `distilgpt2` | 82M | Fastest option, great for quick runs |
| `EleutherAI/gpt-neo-125m` | 125M | Open-source GPT-3 style alternative |
| `microsoft/phi-2` | 2.7B | Strong quality, requires High-RAM runtime |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | LLaMA architecture, efficient on Colab |

---

## File Structure

```
.
├── GPT2_FineTuning_Lab.ipynb   # Main Colab notebook
└── README.md                   # This file
```

---

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [GPT-2 Model Card](https://huggingface.co/gpt2)
- [Hugging Face Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
