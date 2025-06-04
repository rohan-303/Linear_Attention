# Efficient Attention in Transformers: Linear Self-Attention for Machine Translation

This project implements and evaluates a custom **Linear Transformer** model for **English-to-Hindi machine translation**, designed to improve computational efficiency over the standard **Vanilla Transformer**. By reducing the quadratic complexity of self-attention to linear time using kernel-based feature maps, the model achieves faster convergence and lower memory consumption without significantly sacrificing translation quality.

---

## Project Structure
Linear_Attention/
│── Notebook/            
│   ├── data_cleaning.ipynb                   # This notebook has cleaning code for ICD raw dataset
│   ├── data_collection&Analysis.ipynb        # This notebook has the code for Dataset Extraction and EDA on Cleaned Dataset.
│   ├── Evaluation_Desc_ICD.ipynb             # This notebook has the graphs of both training and tetsing data for all models used in Description to ICD Code conversion
│   ├── Evaluation_ICD_Desc.ipynb             # This notebook has the graphs of both training and tetsing data for all models used in ICD Code to Description conversion
│   ├── Testing_Desc_ICD.ipynb                # This notebook has the code for using the best model to generate ICD Code from Description from every models used.
│   ├── Testing_ICD_Desc.ipynb                # This notebook has the code for using the best model to generate Description from Desction from every models used.
│   ├── Tokenization.ipynb                    # This notebook has the code for using BPE tokenizer and adding padding before using Seq2Seq models.
│
│── src/                            
│   ├── main/                      
│   │   ├── Desc_to_ICD/               
│   │   │   ├── BiGRU.py                      # This python file has the code of implementing Bidirectional GRU model along with optuna hyperparameter tuning.
│   │   │   ├── BiLSTM.py                     # This python file has the code of implementing Bidirectional LSTM model along with optuna hyperparameter tuning.
│   │   │   ├── BiRNN.py                      # This python file has the code of implementing Bidirectional RNN model along with optuna hyperparameter tuning.
│   │   │   ├── DeepGRU.py                    # This python file has the code of implementing Deep GRU model along with optuna hyperparameter tuning.
│   │   │   ├── DeepLSTM.py                   # This python file has the code of implementing Deep LSTM model along with optuna hyperparameter tuning.
│   │   │   ├── DeepRNN.py                    # This python file has the code of implementing Deep RNN model along with optuna hyperparameter tuning.
│   │   │   ├── GRU_first.py                  # This python file has the code of implementing GRU model with first 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── GRU_last.py                   # This python file has the code of implementing GRU model with last 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── GRU_middle.py                 # This python file has the code of implementing GRU model with middle 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── LSTM_last.py                  # This python file has the code of implementing LSTM model with last 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── LSTM_first.py                 # This python file has the code of implementing LSTM model with first 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── LSTM_middle.py                # This python file has the code of implementing LSTM model with middle 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── RNN_first.py                  # This python file has the code of implementing RNN model with first 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── RNN_last.py                   # This python file has the code of implementing RNN model with last 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── RNN_middle.py                 # This python file has the code of implementing RNN model with middle 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── Transfomer.py                 # This python file has the code of implementing Transformers model along with optuna hyperparameter tuning.
│   │   ├── ICD_to_Desc/
│   │   │   ├── BiGRU.py                      # This python file has the code of implementing Bidirectional GRU model along with optuna hyperparameter tuning.
│   │   │   ├── BiLSTM.py                     # This python file has the code of implementing Bidirectional LSTM model along with optuna hyperparameter tuning.
│   │   │   ├── BiRNN.py                      # This python file has the code of implementing Bidirectional RNN model along with optuna hyperparameter tuning.
│   │   │   ├── DeepGRU.py                    # This python file has the code of implementing Deep GRU model along with optuna hyperparameter tuning.
│   │   │   ├── DeepLSTM.py                   # This python file has the code of implementing Deep LSTM model along with optuna hyperparameter tuning.
│   │   │   ├── DeepRNN.py                    # This python file has the code of implementing Deep RNN model along with optuna hyperparameter tuning.
│   │   │   ├── GRU.py                        # This python file has the code of implementing GRU model along with optuna hyperparameter tuning.
│   │   │   ├── LSTM.py                       # This python file has the code of implementing LSTM model along with optuna hyperparameter tuning.
│   │   │   ├── RNN.py                        # This python file has the code of implementing RNN model along with optuna hyperparameter tuning.
│   │   │   ├── Transfomer.py                 # This python file has the code of implementing Transformers model along with optuna hyperparameter tuning.
│   │
├── Model_rnn.py                              # This python file has the code of all seq2seq models used along with the train function
├── Transformer_model.py                      # This python file has the code of all modules in transformer architecture used along with the train function
├── utils.py                                  # This python file has the supported dataset function which was used before passing into the model
│── requirements.txt                          # List of required Python libraries
│── README.md                                 # Project documentation

```

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
