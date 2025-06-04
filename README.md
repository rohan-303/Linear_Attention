# Efficient Attention in Transformers: Linear Self-Attention for Machine Translation

This project implements and evaluates a custom **Linear Transformer** model for **English-to-Hindi machine translation**, designed to improve computational efficiency over the standard **Vanilla Transformer**. By reducing the quadratic complexity of self-attention to linear time using kernel-based feature maps, the model achieves faster convergence and lower memory consumption without significantly sacrificing translation quality.

---

## Project Structure


---

## Problem Statement

Traditional Transformers are computationally expensive due to their **O(n²)** attention mechanism. This project investigates the use of **Linear Attention** (O(n) complexity) as a more scalable alternative, especially for long input sequences in machine translation tasks.

---

## Dataset and Preprocessing

- **Corpus:** English-Hindi parallel dataset (sampled from HuggingFace, ~10% subset)
- **Tokenization:** Byte Pair Encoding (BPE), vocabulary size of 50,000
- **Special Tokens:** `[PAD]`, `[UNK]`, `[SOS]`, `[EOS]`
- **Max Sequence Length:** 128 tokens
- **Input Formatting:**
  - **Encoder Input:** `[SOS] + tokens + [EOS] + PADs`
  - **Decoder Input:** `[SOS] + tokens + PADs`
  - **Target Labels:** `tokens + [EOS] + PADs`

---

##  Models Implemented

| Model Variant        | Attention Mechanism       | Complexity | Softmax Used |
|----------------------|----------------------------|------------|--------------|
| Vanilla Transformer  | Scaled Dot-Product         | O(n²)      | Yes       |
| Linear Transformer   | Feature-Based Kernelized   | O(n)       | No        |

---

## Training Configuration

- **Loss Function:** CrossEntropyLoss (ignoring `[PAD]`)
- **Optimizer:** Adam with weight decay (`1e-5`)
- **Batch Size:** 32
- **Dropout:** 0.1–0.5 (tuned)
- **Gradient Clipping:** Max norm = 5
- **Epochs:** 100 (Linear), 250 (Vanilla)
- **Scheduler:** Cosine Annealing (optional)

---

## Hyperparameter Tuning (Optuna)

| Hyperparameter        | Search Space            |
|-----------------------|-------------------------|
| Learning Rate         | [0.0001, 0.001]          |
| Model Dimension       | {64, 128, 256}           |
| Feedforward Width     | {256, 512, 1024}         |
| Encoder/Decoder Layers| {2, 4}                   |
| Attention Heads       | {2, 4}                   |
| Dropout Rate          | [0.1, 0.5]               |

**Objective:** Minimize validation loss over 5–10 trials per model.

---

## Evaluation Metrics

- **Token-Level:** Accuracy, Precision, Recall, F1-Score
- **Sequence-Level:** Word Error Rate (WER), Character Error Rate (CER)

| Metric     | Description                                   |
|------------|-----------------------------------------------|
| Accuracy   | % of correctly predicted tokens               |
| Precision  | TP / (TP + FP)                                |
| Recall     | TP / (TP + FN)                                |
| F1-Score   | Harmonic mean of Precision and Recall         |
| CER        | Edit distance at character level              |
| WER        | Edit distance at word level                   |

---

## Results

### Scenario: English → Hindi Translation

| Metric            | Vanilla Transformer | Linear Transformer |
|-------------------|---------------------|--------------------|
| Accuracy          | 0.38                | **0.987**          |
| Precision         | 0.18                | **0.73**           |
| Recall            | 0.16                | **0.75**           |
| F1-Score          | 0.15                | **0.74**           |
| Loss              | 4.1                 | **0.18**           |
| CER               | 1% (stable)         | 10–60% → stabilized |
| WER               | 4% (stable)         | 28% → stabilized   |

---

## Conclusion

The **Linear Transformer** achieves significant improvements in **efficiency** while delivering **competitive translation performance** compared to the traditional Transformer. It converges faster, uses less memory, and is better suited for long sequences, making it an ideal candidate for real-world NLP tasks with limited resources.

---

## References

- Vaswani et al., *“Attention is All You Need”*, NeurIPS 2017  
- Katharopoulos et al., *“Transformers are RNNs”*, ICML 2020  
- Wang et al., *“Linformer: Self-Attention with Linear Complexity”*, 2020  
- Choromanski et al., *“Rethinking Attention with Performers”*, ICLR 2021  
- Akiba et al., *“Optuna: A Hyperparameter Optimization Framework”*, KDD 2019

---
