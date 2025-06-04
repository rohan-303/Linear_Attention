import torch
import torch.nn as nn
import math
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerTrainEval:
    def __init__(self, num_epochs, train_loader, val_loader, lr, data_dir, save_ckpt_dir,
                 vocab_size, seq_len, tokenizer_src, tokenizer_tgt, writer: str):
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        self.data_dir = data_dir
        self.save_ckpt_dir = save_ckpt_dir
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.writer = SummaryWriter(writer)

    def train(self, model):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)

        best_val_loss = float('inf')
        best_epoch = 0
        train_loss_list = []
        train_accuracy_list = []
        train_precision_list = []
        train_recall_list = []
        train_f1_list = []
        val_loss_list = []

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            model.train()
            total_loss = 0
            all_preds, all_targets = [], []

            for batch in self.train_loader:
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)
                label = batch['label'].to(device)

                optimizer.zero_grad()

                encoder_output = model.encode(encoder_input, encoder_mask)
                #print(f"The out encoder shape {encoder_output.shape}")
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                #print(f"The out shape of decoder {decoder_output.shape}")
                proj_output = model.project(decoder_output)
                #print(f"The shape of Projection Output is: {proj_output.shape}")

                loss = self.loss_fn(proj_output.reshape(-1, self.vocab_size), label.reshape(-1))
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()

                total_loss += loss.item()

                preds = proj_output.argmax(dim=-1)
                mask = label != 0
                all_preds.append(preds[mask].detach().cpu())
                all_targets.append(label[mask].detach().cpu())

            avg_loss = total_loss / len(self.train_loader)
            precision, recall, f1, _ = precision_recall_fscore_support(
                torch.cat(all_targets), torch.cat(all_preds), average='macro', zero_division=0)
            accuracy = accuracy_score(torch.cat(all_targets), torch.cat(all_preds))

            train_loss_list.append(avg_loss)
            train_accuracy_list.append(accuracy)
            train_precision_list.append(precision)
            train_recall_list.append(recall)
            train_f1_list.append(f1)

            print(f"Train Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

            if epoch % 5 == 0 or epoch == self.num_epochs:
                val_avg_loss, val_acc, val_prec, val_rec, val_f1 = self.evaluate(model)
                val_loss_list.append(val_avg_loss)
                print(f"Val Loss: {val_avg_loss:.4f} | Acc: {val_acc:.4f} | P: {val_prec:.4f} | R: {val_rec:.4f} | F1: {val_f1:.4f}")
                self.run_validation(model, global_step=epoch)

                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    best_epoch = epoch
                    self.checkpoint(model)

        print(f"\nBest model saved from epoch {best_epoch} with val loss {best_val_loss:.4f}")
        return train_loss_list, val_loss_list, train_accuracy_list, train_precision_list, train_recall_list, train_f1_list, best_val_loss

    def evaluate(self, model):
        model.eval()
        total_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                label = batch['label'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)

                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)

                loss = self.loss_fn(proj_output.reshape(-1, self.vocab_size), label.reshape(-1))
                total_loss += loss.item()

                preds = proj_output.argmax(dim=-1)
                mask = label != 0
                all_preds.append(preds[mask].detach().cpu())
                all_targets.append(label[mask].detach().cpu())

        avg_loss = total_loss / len(self.val_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(
            torch.cat(all_targets), torch.cat(all_preds), average='macro', zero_division=0)
        accuracy = accuracy_score(torch.cat(all_targets), torch.cat(all_preds))
        return avg_loss, accuracy, precision, recall, f1

    def run_validation(self, model, global_step, num_examples=2):
        from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

        model.eval()
        count = 0
        source_texts, expected, predicted = [], [], []

        with torch.no_grad():
            for batch in self.val_loader:
                if count == num_examples:
                    break

                encoder_input = batch["encoder_input"].to(device)
                encoder_mask = batch["encoder_mask"].to(device)

                assert encoder_input.size(0) == 1

                decoded = greedy_decode(
                    model, encoder_input, encoder_mask, self.tokenizer_src, self.tokenizer_tgt, self.seq_len, device)

                source_texts.append(batch["src_text"][0])
                expected.append(batch["tgt_text"][0])
                predicted.append(self.tokenizer_tgt.decode(decoded.detach().cpu().numpy()))

                print("-" * 100)
                print(f"{'SOURCE:':>12} {source_texts[-1]}")
                print(f"{'TARGET:':>12} {expected[-1]}")
                print(f"{'PREDICTED:':>12} {predicted[-1]}")

                count += 1

        if self.writer:
            self.writer.add_scalar('validation/cer', CharErrorRate()(predicted, expected), global_step)
            self.writer.add_scalar('validation/wer', WordErrorRate()(predicted, expected), global_step)
            self.writer.add_scalar('validation/bleu', BLEUScore()(predicted, expected), global_step)
            self.writer.flush()

    def checkpoint(self, model):
        model_out_path = f"{self.save_ckpt_dir}/best_model.pth"
        torch.save(model.state_dict(), model_out_path)
        print(f"Model checkpoint saved to {model_out_path}")
