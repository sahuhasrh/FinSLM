# FinancialGPT — 22M-Parameter Decoder-Only Transformer from Scratch

Built a GPT-style language model entirely in raw PyTorch — no HuggingFace Trainer,
no prebuilt transformer classes — trained on a 5-dataset financial corpus (~8M tokens).

![Training Curves](assets/training_curves.png)

## Architecture
| Component       | Detail                |
|----------------|-----------------------|
| Type           | Decoder-only (causal) |
| Parameters     | ~22M                  |
| Embed dim      | 384                   |
| Layers / Heads | 8 / 8                 |
| Context length | 256 tokens            |
| Vocab          | 16,384 (custom BPE)   |

## Built From Scratch
- Causal self-attention with manual upper-triangular mask
- Pre-LayerNorm transformer blocks
- Weight tying (embedding ↔ output projection)
- Custom BPE tokenizer (16K vocab)
- Mixed precision AMP FP16 training loop
- Cosine LR with linear warmup
- Top-K sampling + repetition penalty (1.5)

## Dataset Pipeline
5 datasets → cleaned → ~174K sentences → ~8M BPE tokens
Financial PhraseBank · FiQA · Finance Alpaca · FinGPT · Financial News

## Results
| Metric       | Value |
|-------------|-------|
| Best Val PPL | XX.XX |   ← fill this in from your training_history.json
| Best Val Loss| X.XXXX|   ← fill this in

## LR Schedule
![LR Schedule](assets/lr_schedule.png)

## Temperature Diversity Analysis
![Temperature Diversity](assets/temperature_diversity.png)

## Run on Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/FinancialGPT/blob/main/FinancialGPT_v3.ipynb)

## Requirements
```
torch>=2.0
tokenizers
datasets==2.14.7
matplotlib
seaborn
```