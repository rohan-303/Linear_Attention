import torch
import pandas as pd
import os
import sys
import random
import numpy as np
import optuna
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import *
from model import *
from train import *

data_dir = '/home/careinfolab/Dr_Luo/Rohan/Linear_Attention/results/Vanilla_Transformer'
checkpoint_dir = '/home/careinfolab/Dr_Luo/Rohan/Linear_Attention/results/Vanilla_Transformer/checkpoints'
results_file = '/home/careinfolab/Dr_Luo/Rohan/Linear_Attention/results/Vanilla_Transformer/results.txt'

os.makedirs(checkpoint_dir, exist_ok=True)
with open(results_file, "w") as file:
    file.write("")

en_tokenizer = Tokenizer.from_file("/home/careinfolab/Dr_Luo/Rohan/Linear_Attention/Notebook/custom_en_tokenizer.json")
hi_tokenizer = Tokenizer.from_file("/home/careinfolab/Dr_Luo/Rohan/Linear_Attention/Notebook/custom_hi_tokenizer.json")

df = pd.read_csv("/home/careinfolab/Dr_Luo/Rohan/Linear_Attention/Dataset/eng_hindi_cleaned.csv")

seq_len = 128

df = df[
    df['english_sentence'].apply(lambda x: isinstance(x, str)) &
    df['hindi_sentence'].apply(lambda x: isinstance(x, str))
].dropna(subset=['english_sentence', 'hindi_sentence']).reset_index(drop=True)

print("Tokenizing and computing lengths...", flush=True)
df['en_len'] = [len(en_tokenizer.encode(x).ids) for x in tqdm(df['english_sentence'], desc="English")]
df['hi_len'] = [len(hi_tokenizer.encode(x).ids) for x in tqdm(df['hindi_sentence'], desc="Hindi")]

df = df[(df['en_len'] + 2 <= seq_len) & (df['hi_len'] + 1 <= seq_len)].reset_index(drop=True)

if df.empty:
    raise ValueError("No data left after filtering long sequences. Adjust seq_len or check dataset.")

print(f"Filtered dataset: {len(df)} rows after removing sequences > {seq_len} tokens.", flush=True)

train_data, dummy_data = train_test_split(df, test_size=0.3, random_state=42)
test_data, val_data = train_test_split(dummy_data, test_size=2/3, random_state=42)

train_dataset = TranslationDataset(train_data, en_tokenizer, hi_tokenizer, 'english_sentence', 'hindi_sentence', seq_len)
val_dataset = TranslationDataset(val_data, en_tokenizer, hi_tokenizer, 'english_sentence', 'hindi_sentence', seq_len)
test_dataset = TranslationDataset(test_data, en_tokenizer, hi_tokenizer, 'english_sentence', 'hindi_sentence', seq_len)


def objective(trial):
    print(">>> Starting trial", trial.number)
    lr = trial.suggest_float('lr', 0.0001, 0.001)
    d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512])
    d_ff = trial.suggest_categorical('d_ff', [256, 512, 1024, 2048])
    N = trial.suggest_categorical('N', [2, 4, 6, 8])
    h = trial.suggest_categorical('h', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    num_epochs = trial.suggest_categorical('num_epochs', [250])

    print(">>> Loading dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=32,num_workers=4,pin_memory=True,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1,num_workers=2,pin_memory=True,shuffle=False)   

    print(">>> Building model...")
    model = build_transformer(
        src_vocab_size=en_tokenizer.get_vocab_size(),
        tgt_vocab_size=hi_tokenizer.get_vocab_size(),
        src_seq_len=seq_len,
        tgt_seq_len=seq_len,
        d_model=d_model,
        N=N,
        h=h,
        dropout=dropout,
        d_ff=d_ff,
        use_Linear=False
    )

    print(">>> Training model...")
    trainer = TransformerTrainEval(
        num_epochs=num_epochs,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        lr=lr,
        data_dir=data_dir,
        save_ckpt_dir=checkpoint_dir,
        vocab_size=hi_tokenizer.get_vocab_size(),
        seq_len=seq_len,
        tokenizer_src=en_tokenizer,
        tokenizer_tgt=hi_tokenizer,
        writer='runs/vanilla_transformer'
    )

    results = trainer.train(model)
    best_val_loss = results[-1]

    del model
    torch.cuda.empty_cache()

    print(">>> Trial complete.")
    return best_val_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

print("Best parameters:", study.best_params)
print("Best C-index:", study.best_value)

best_params = study.best_params
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4,pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=1,num_workers=2,pin_memory=True,shuffle=False)

model = build_transformer(
    src_vocab_size=en_tokenizer.get_vocab_size(),
    tgt_vocab_size=hi_tokenizer.get_vocab_size(),
    src_seq_len=seq_len,
    tgt_seq_len=seq_len,
    d_model=best_params['d_model'],
    N=best_params['N'],
    h=best_params['h'],
    dropout=best_params['dropout'],
    d_ff=best_params['d_ff'],
    use_Linear=False
)

trainer = TransformerTrainEval(
    num_epochs=best_params['num_epochs'],
    train_loader=train_dataloader,
    val_loader=test_dataloader,
    lr=best_params['lr'],
    data_dir=data_dir,
    save_ckpt_dir=checkpoint_dir,
    vocab_size=hi_tokenizer.get_vocab_size(),
    seq_len=seq_len,
    tokenizer_src=en_tokenizer,
    tokenizer_tgt=hi_tokenizer,
    writer='runs/vanilla_transformer'
)


train_loss, val_avg_loss, train_accuracy, train_precision, train_recall, train_f1score, test_accuracy, test_precision, test_recall, test_f1score,best_val_loss = trainer.train(model)

results = {
    'best params': best_params,
    'train_loss': train_loss,
    'test_loss': val_avg_loss,
    'train_accuracy': train_accuracy,
    'train_precision': train_precision,
    'train_recall': train_recall,
    'train_f1': train_f1score,
    'test_accuracy': test_accuracy,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1score,
    'total_parameters': count_parameters(model)
}

torch.save(train_loss, f"{data_dir}/train_loss.pth")
torch.save(val_avg_loss, f"{data_dir}/test_loss.pth")
torch.save(train_accuracy, f"{data_dir}/train_accuracy.pth")
torch.save(train_precision, f"{data_dir}/train_precision.pth")
torch.save(train_recall, f"{data_dir}/train_recall.pth")
torch.save(train_f1score, f"{data_dir}/train_f1score.pth")
torch.save(test_accuracy, f"{data_dir}/test_accuracy.pth")
torch.save(test_precision, f"{data_dir}/test_precision.pth")
torch.save(test_recall, f"{data_dir}/test_recall.pth")
torch.save(test_f1score, f"{data_dir}/test_f1score.pth")

with open(results_file, "a") as file:
    for key, value in results.items():
        file.write(f"{key}: {value}\n")
    file.write("\n" + "-" * 50 + "\n\n")
