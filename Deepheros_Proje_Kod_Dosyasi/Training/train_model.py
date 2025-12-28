import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.textcnn import TextCNN
from models.bigru import BiGRU
from models.bilstm_attention import BiLSTMAttention
from models.transformer_encoder import TransformerEncoderClassifier
from utils.artifact_utils import save_all_artifacts

MODEL_CONFIGS = {
    'A_BiLSTM_Attn': {
        'type': 'pytorch',
        'model_class': BiLSTMAttention,
        'hyperparams': {
            'embed_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.5,
            'learning_rate': 0.001,
            'num_epochs': 8,
            'batch_size': 64
        }
    },
    'A_DistilBERT': {
        'type': 'transformer',
        'model_id': 'distilbert-base-uncased',
        'hyperparams': {
            'learning_rate': 1e-4,
            'num_epochs': 3,
            'batch_size': 8,
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'use_lora': True
        }
    },
    'B_TextCNN': {
        'type': 'pytorch',
        'model_class': TextCNN,
        'hyperparams': {
            'embed_dim': 128,
            'num_filters': 100,
            'filter_sizes': [3, 4, 5],
            'dropout': 0.5,
            'learning_rate': 0.001,
            'num_epochs': 15,
            'batch_size': 32
        }
    },
    'B_BERTTiny': {
        'type': 'transformer',
        'model_id': 'prajjwal1/bert-tiny',
        'hyperparams': {
            'learning_rate': 1e-4,
            'num_epochs': 3,
            'batch_size': 8,
            'lora_r': 4,
            'lora_alpha': 8,
            'lora_dropout': 0.1,
            'freeze_encoder': True,
            'use_lora': True
        }
    },
    'C_BiGRU': {
        'type': 'pytorch',
        'model_class': BiGRU,
        'hyperparams': {
            'embed_dim': 128,
            'hidden_dim': 128,
            'num_layers': 1,
            'dropout': 0.5,
            'learning_rate': 0.001,
            'num_epochs': 3,
            'batch_size': 128
        }
    },
    'C_TransformerEncoder': {
        'type': 'pytorch',
        'model_class': TransformerEncoderClassifier,
        'hyperparams': {
            'embed_dim': 256,
            'num_heads': 8,
            'num_layers': 3,
            'hidden_dim': 512,
            'dropout': 0.1,
            'learning_rate': 0.0001,
            'num_epochs': 6,
            'batch_size': 32
        }
    }
}

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_length=512, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {k: v.squeeze(0) for k, v in encoded.items()}, torch.tensor(label, dtype=torch.long)
        else:
            tokens = text.lower().split()
            token_ids = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens]

            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                token_ids = token_ids + [self.vocab.get('<PAD>', 0)] * (self.max_length - len(token_ids))

            return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def build_vocab(texts, min_freq=2):
    word_counts = Counter()
    for text in texts:
        tokens = text.lower().split()
        word_counts.update(tokens)

    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab

def load_label_mapping(artifacts_dir):
    mapping_path = os.path.join(artifacts_dir, 'label_mapping.json')
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            return json.load(f)
    return None

def train_pytorch_model(model_key, config, train_df, val_df, test_df, artifact_dir, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building vocabulary...")
    vocab = build_vocab(train_df['question'].tolist(), min_freq=2)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    num_classes = len(train_df['label'].unique())

    max_length = 512
    train_dataset = TextDataset(train_df['question'].tolist(), train_df['label'].tolist(), vocab, max_length)
    val_dataset = TextDataset(val_df['question'].tolist(), val_df['label'].tolist(), vocab, max_length)
    test_dataset = TextDataset(test_df['question'].tolist(), test_df['label'].tolist(), vocab, max_length)

    batch_size = config['hyperparams']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_class = config['model_class']
    model_params = {
        'vocab_size': vocab_size,
        'num_classes': num_classes
    }

    if model_key == 'C_TransformerEncoder':
        model_params.update({
            'embed_dim': config['hyperparams']['embed_dim'],
            'num_heads': config['hyperparams']['num_heads'],
            'num_layers': config['hyperparams']['num_layers'],
            'hidden_dim': config['hyperparams']['hidden_dim'],
            'dropout': config['hyperparams']['dropout']
        })
    elif model_key == 'A_BiLSTM_Attn':
        model_params.update({
            'embed_dim': config['hyperparams']['embed_dim'],
            'hidden_dim': config['hyperparams']['hidden_dim'],
            'num_layers': config['hyperparams']['num_layers'],
            'dropout': config['hyperparams']['dropout']
        })
    elif model_key == 'B_TextCNN':
        model_params.update({
            'embed_dim': config['hyperparams']['embed_dim'],
            'num_filters': config['hyperparams']['num_filters'],
            'filter_sizes': config['hyperparams']['filter_sizes'],
            'dropout': config['hyperparams']['dropout'],
            'max_length': max_length
        })
    else:
        model_params.update({
            'embed_dim': config['hyperparams']['embed_dim'],
            'hidden_dim': config['hyperparams']['hidden_dim'],
            'num_layers': config['hyperparams']['num_layers'],
            'dropout': config['hyperparams']['dropout']
        })

    model = model_class(**model_params)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['hyperparams']['learning_rate'])

    num_epochs = config['hyperparams']['num_epochs']
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_acc = 0
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        model.train()
        train_loss = 0
        for texts, labels in tqdm(train_loader, desc="Training"):
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for texts, labels in tqdm(val_loader, desc="Validating"):
                texts, labels = texts.to(device), labels.to(device)

                logits = model(texts)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_acc)

        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'config': config['hyperparams'],
                'model_key': model_key,
                'vocab_size': vocab_size,
                'num_classes': num_classes
            }
            torch.save(checkpoint, os.path.join(artifact_dir, "best_model.pt"))
            print(f"Saved best model (val_acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered (no improvement for {patience} epochs)")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break

    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(artifact_dir, "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_preds = []
    test_probs = []
    test_labels = []

    with torch.no_grad():
        for texts, labels in tqdm(test_loader, desc="Testing"):
            texts = texts.to(device)
            logits = model(texts)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels.numpy())

    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)

    run_config = {
        'model_key': model_key,
        'seed': seed,
        'hyperparameters': config['hyperparams'],
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        'library_versions': {
            'torch': torch.__version__,
            'numpy': np.__version__
        }
    }

    save_all_artifacts(
        test_labels, test_preds, test_probs,
        ids=test_df.get('id', range(len(test_labels))).tolist(),
        history=history,
        config=run_config,
        class_names=None,
        artifact_dir=artifact_dir
    )

    return model

def train_transformer_model(model_key, config, train_df, val_df, test_df, artifact_dir, seed):
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        TrainingArguments, Trainer, set_seed
    )
    from transformers.trainer_callback import EarlyStoppingCallback
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from sklearn.metrics import accuracy_score, f1_score

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    model_id = config['model_id']
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    def tokenize_function(examples):
        return tokenizer(
            examples["question"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    num_classes = len(train_df['label'].unique())

    if device.type == "cpu":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_classes, token=hf_token, torch_dtype=torch.float32
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_classes, token=hf_token
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    if config['hyperparams'].get('freeze_encoder', False):
        for param in model.base_model.parameters():
            param.requires_grad = False
        print("Frozen encoder layers for faster training")

    if config['hyperparams'].get('use_lora', True):
        target_modules = ['q_lin', 'k_lin', 'v_lin', 'out_lin'] if 'distilbert' in model_id.lower() else ['query', 'key', 'value', 'dense']
        lora_config = LoraConfig(
            r=config['hyperparams']['lora_r'],
            lora_alpha=config['hyperparams']['lora_alpha'],
            target_modules=target_modules,
            lora_dropout=config['hyperparams']['lora_dropout'],
            bias="none",
            task_type="SEQ_CLS"
        )
        model = get_peft_model(model, lora_config)

    model = model.to(device)

    training_args = TrainingArguments(
        output_dir=os.path.join(artifact_dir, "checkpoints"),
        learning_rate=config['hyperparams']['learning_rate'],
        num_train_epochs=config['hyperparams']['num_epochs'],
        per_device_train_batch_size=config['hyperparams']['batch_size'],
        per_device_eval_batch_size=config['hyperparams']['batch_size'] * 2,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(artifact_dir, "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        save_total_limit=3,
        fp16=False if device.type == "cpu" else True,
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_micro": f1_score(labels, predictions, average='micro'),
            "f1_macro": f1_score(labels, predictions, average='macro'),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("Starting training...")
    trainer.train()

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1_micro': []
    }

    for log in trainer.state.log_history:
        if 'loss' in log and 'eval_loss' not in log:
            history['train_loss'].append(log['loss'])
        if 'eval_loss' in log:
            history['val_loss'].append(log['eval_loss'])
        if 'eval_accuracy' in log:
            history['val_accuracy'].append(log['eval_accuracy'])
        if 'eval_f1_micro' in log:
            history['val_f1_micro'].append(log['eval_f1_micro'])

    print("\nEvaluating on test set...")
    test_predictions = trainer.predict(test_dataset)
    test_logits = test_predictions.predictions
    test_preds = np.argmax(test_logits, axis=-1)
    test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_labels = test_df['label'].values

    best_model_path = os.path.join(artifact_dir, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    run_config = {
        'model_key': model_key,
        'model_id': model_id,
        'seed': seed,
        'hyperparameters': config['hyperparams'],
        'num_classes': num_classes,
        'checkpoint_path': best_model_path,
        'library_versions': {
            'torch': torch.__version__
        }
    }

    save_all_artifacts(
        test_labels, test_preds, test_probs,
        ids=test_df.get('id', range(len(test_labels))).tolist(),
        history=history,
        config=run_config,
        class_names=None,
        artifact_dir=artifact_dir
    )

    return model

def main():
    parser = argparse.ArgumentParser(description="Train a single model")
    parser.add_argument('--model_key', type=str, required=True,
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model key (e.g., A_BiLSTM_Attn)')
    parser.add_argument('--data', type=str, default='./data/train.csv',
                       help='Path to training data CSV')
    parser.add_argument('--splits', type=str, default='./artifacts/splits',
                       help='Directory containing split indices')
    parser.add_argument('--out', type=str, default='./artifacts',
                       help='Output directory for artifacts')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    if 'Question' in df.columns:
        df = df.rename(columns={'Question': 'question'})

    print(f"Loading split indices from: {args.splits}")
    train_idx = np.load(os.path.join(args.splits, 'train_idx.npy'))
    val_idx = np.load(os.path.join(args.splits, 'val_idx.npy'))
    test_idx = np.load(os.path.join(args.splits, 'test_idx.npy'))

    label_mapping = load_label_mapping(args.out)
    if label_mapping and df['label'].dtype == 'object':
        label_to_id = {k: int(v) for k, v in label_mapping['label_to_id'].items()}
        df['label'] = df['label'].map(label_to_id)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")

    config = MODEL_CONFIGS[args.model_key]
    artifact_dir = os.path.join(args.out, args.model_key)
    os.makedirs(artifact_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Training: {args.model_key}")
    print(f"{'='*70}\n")

    if config['type'] == 'pytorch':
        train_pytorch_model(args.model_key, config, train_df, val_df, test_df, artifact_dir, args.seed)
    else:
        train_transformer_model(args.model_key, config, train_df, val_df, test_df, artifact_dir, args.seed)

    print(f"\n✓ Training completed for {args.model_key}")

if __name__ == "__main__":
    main()
