# FinancialGPT — Decoder-Only Transformer Built from Scratch

A **20.6M-parameter GPT-style language model** trained on financial text, implemented entirely in raw PyTorch — no HuggingFace Trainer, no prebuilt transformer classes. Every component — attention, tokenizer, training loop — written from scratch.

---

## What Makes This Different

Most NLP projects fine-tune a pretrained model. This one builds the entire stack from first principles:

- **Custom BPE tokenizer** — trained from scratch on the financial corpus (16K vocab), not borrowed from GPT-2
- **Causal self-attention** — implemented manually with upper-triangular masking, not via `nn.MultiheadAttention`
- **Training loop** — mixed precision, gradient clipping, cosine LR, early stopping — all hand-written
- **Inference** — greedy, temperature sampling, Top-K, and repetition penalty implemented from scratch

---

## Architecture

| Component        | Detail                        |
|-----------------|-------------------------------|
| Type            | Decoder-only (causal) transformer |
| Parameters      | 20.6M                         |
| Embedding dim   | 384                           |
| Transformer layers | 8                          |
| Attention heads | 8                             |
| FFN multiplier  | 4×                            |
| Context length  | 256 tokens                    |
| Vocab size      | 16,384 (custom BPE)           |
| Position encoding | Learned                    |

**Parameter breakdown:**
- Embeddings: 6.39M
- Attention layers: 4.72M
- FFN layers: 9.45M
- Output head: shared with token embeddings (weight tying)

---

## Dataset & Tokenizer

**5 financial datasets → ~174K sentences → ~11.6M BPE tokens**

| Dataset | Content |
|---------|---------|
| Financial PhraseBank | Sentiment-labelled financial sentences |
| FiQA | Financial question-answer pairs |
| Finance Alpaca | Financial instruction-response pairs |
| FinGPT Sentiment | Financial news with sentiment labels |
| Financial News Classification | News articles across financial categories |

**Preprocessing pipeline:**
- Removed URLs, Twitter handles, hashtags, HTML entities
- Filtered sentences < 8 words or > 200 words
- Removed high digit-ratio text (ticker spam)
- Final corpus: 174,001 clean sentences

**BPE Tokenizer** trained via HuggingFace `tokenizers` library on the financial corpus — special tokens `[BOS]`, `[EOS]`, `[PAD]`, `[UNK]`. Financial subwords like `ĠEBITDA`, `Ġbasis`, `ĠFed` get clean encodings.

---

## Training Setup

| Hyperparameter   | Value         |
|-----------------|---------------|
| Optimizer       | AdamW         |
| Learning rate   | 2e-4          |
| Weight decay    | 0.1           |
| β values        | (0.9, 0.95)   |
| LR schedule     | Linear warmup (100 steps) → Cosine decay |
| Gradient clip   | 1.0           |
| Batch size      | 32            |
| Max epochs      | 30            |
| Early stopping  | Patience = 6  |
| Precision       | AMP FP16 (mixed precision) |
| Hardware        | Google Colab T4 GPU (~2–3 hrs) |

Key stability choices: **Pre-LayerNorm** (vs post-LN) for stable gradient flow, **weight tying** between token embedding and output projection to reduce parameters.

---

## Results

| Metric          | Value   |
|----------------|---------|
| Best Val Loss  | 3.1061  |
| Best Val PPL   | **22.33** |
| Final Train Loss | 3.07  |
| Training sentences | 174,001 |
| Total tokens   | ~11.6M  |

Perplexity dropped from ~110 at epoch 3 down to **22.33** at convergence — consistent improvement across all epochs with no overfitting (train/val curves stayed close).

---

## Sample Outputs

**Prompt:** `The Federal Reserve raised interest rates`
> The Federal Reserve raised interest rates to higher levels, as the Fed will adjust its monetary policy for the next couple of years.

**Prompt:** `Goldman Sachs reported quarterly earnings`
> Goldman Sachs reported quarterly earnings results for the first quarter of 2008, up from $320 million to $6.27 billion in revenue.

**Prompt:** `Inflation concerns have led investors`
> Inflation concerns have led investors to buy a dip in the stock market.

**Prompt:** `The stock market declined sharply after`
> The stock market declined sharply after the news, with major indices falling more than 3% in early trading.

---

## Key Implementation Details

**Attention:**
```python
# Manual causal mask — no nn.MultiheadAttention
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

**Weight tying:**
```python
# Output projection shares weights with token embedding
self.head.weight = self.tok_emb.weight
```

**Repetition penalty** at inference — tokens already generated get their logits divided by 1.5, reducing loops and repetitive output.

**Mixed precision:**
```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    logits, loss = model(x, targets)
scaler.scale(loss).backward()
```

---

## Project Structure

```
FinSLM/
├── Code.ipynb                        # Full notebook with EDA, training, evaluation
├── config.json                       # Model + training hyperparameters
├── checkpoints/
│   ├── best_model.pt                 # Best checkpoint by val loss
│   └── latest.pt                     # Latest checkpoint (resume training)
├── tokenizer/
│   └── bpe_financial.json            # Trained BPE tokenizer
├── data/
│   ├── corpus.txt                    # Cleaned financial corpus
│   └── token_tensors.pt              # Pre-tokenized dataset
├── logs/
│   ├── training_history.json         # Loss/PPL per epoch
│   └── generation_samples.json       # Inference outputs
├── plots/
│   ├── training_curves.png
│   ├── lr_schedule.png
│   ├── eda_length_dist.png
│   └── temperature_diversity.png
└── exports/
    ├── final_metrics.json
    ├── resume_bullets.txt
    └── interview_talking_points.txt
```

---

## Run It

```bash
pip install torch>=2.0 tokenizers datasets==2.14.7 matplotlib seaborn pyarrow
```

Open `Code.ipynb` in Google Colab with a T4 GPU. The notebook is structured in 9 sections — you can run them top to bottom or resume from a checkpoint.

---

## Future Work

- Rotary Position Embeddings (RoPE) to replace learned positional encodings
- Flash Attention for memory efficiency at longer context
- Fine-tune on SEC filings / earnings call transcripts
- Scale context to 512+ tokens
- RLHF-style alignment for factually grounded financial outputs
