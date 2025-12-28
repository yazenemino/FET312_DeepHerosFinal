import os
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name, device="cuda", classifier=True, quantization_config=None):

    logger.info(f"Loading model: {model_name}")
    print(f"[INFO] Loading model: {model_name} (this may take 5-10 minutes on CPU)...")

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

    try:
        print("[INFO] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            padding_side="left",
            token=hf_token
        )
        print("[INFO] Tokenizer loaded successfully")

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = "[PAD]"
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        if not classifier:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device if device != "cpu" else None,
                quantization_config=quantization_config,
                dtype="auto",
                trust_remote_code=True,
                token=hf_token,
            )
        else:
            print("[INFO] Loading model weights (this is the slow part on CPU)...")
            if device == "cpu":
                if quantization_config is not None:
                    print("[INFO] Note: Quantization on CPU can be slow, loading without it...")
                    quantization_config = None
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=8,
                    quantization_config=quantization_config,
                    dtype=torch.float32,
                    trust_remote_code=True,
                    token=hf_token,
                )
                model = model.to(device)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=8,
                    quantization_config=quantization_config,
                    dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token,
                )
            print("[INFO] Model weights loaded, moving to device...")

        model.config.pad_token_id = tokenizer.pad_token_id

        logger.info(f"Model loaded successfully: {model_name}")
        print(f"[INFO] Model {model_name} loaded successfully!")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise

def predict_with_classifier(model, tokenizer, questions, device="cuda"):
    model.to(device)
    model.eval()

    predictions = []

    for question in questions:
        inputs = tokenizer(
            question, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()

        predictions.append(predicted_label)

    return predictions 