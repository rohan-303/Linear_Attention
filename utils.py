import torch
from torch.utils.data import Dataset
import torchmetrics
import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class TranslationDataset(Dataset):
    def __init__(self, df, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.df = df
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.pad_token = tokenizer_tgt.encode("[PAD]").ids[0]
        self.sos_token = tokenizer_tgt.encode("[SOS]").ids[0]
        self.eos_token = tokenizer_tgt.encode("[EOS]").ids[0]

        self.data = []

        print("Preprocessing and tokenizing dataset...", flush=True)
        for i in tqdm(range(len(df)), desc="Tokenizing"):
            src_text = df[src_lang].iloc[i]
            tgt_text = df[tgt_lang].iloc[i]

            enc_input_tokens = tokenizer_src.encode(src_text).ids
            dec_input_tokens = tokenizer_tgt.encode(tgt_text).ids

            enc_pad_len = seq_len - len(enc_input_tokens) - 2  
            dec_pad_len = seq_len - len(dec_input_tokens) - 1 

            if enc_pad_len < 0 or dec_pad_len < 0:
                continue  
            encoder_input = [self.sos_token] + enc_input_tokens + [self.eos_token] + [self.pad_token] * enc_pad_len
            decoder_input = [self.sos_token] + dec_input_tokens + [self.pad_token] * dec_pad_len
            label = dec_input_tokens + [self.eos_token] + [self.pad_token] * dec_pad_len
            self.data.append({
                "encoder_input": torch.tensor(encoder_input, dtype=torch.long),
                "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.long),
                "src_text": src_text,
                "tgt_text": tgt_text
            })

        if len(self.data) == 0:
            raise ValueError("All sequences were too long after filtering.")

        print(f"Loaded {len(self.data)} valid examples.", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoder_input = item["encoder_input"]
        decoder_input = item["decoder_input"]
        label = item["label"]
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() 
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(self.seq_len) 

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": item["src_text"],
            "tgt_text": item["tgt_text"]
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.full((1, 1), sos_idx, dtype=torch.long).to(device)

    while True:
        if decoder_input.size(1) >= max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        decode_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        decoder_output = decode_output[0] if isinstance(decode_output, tuple) else decode_output

        prob = model.project(decoder_output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, writer, global_step, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            print("-" * console_width)
            print(f"{'SOURCE: ':>12}{source_text}")
            print(f"{'TARGET: ':>12}{target_text}")
            print(f"{'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                break

    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)

        writer.flush()