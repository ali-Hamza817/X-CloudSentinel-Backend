import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .utils import get_mc_dropout_prediction

class SentinelClassifier:
    def __init__(self, model_path="f:/Fullbright Scholarship/X-CloudSentinel/backend/models/X-CloudSentinel-distilbert"):
        print(f"Loading classifier from {model_path}...")
        if not os.path.exists(model_path):
            print(f"CRITICAL: Classifier model path not found: {model_path}")
        else:
            # Check if it's a directory (Standard Transformers)
            if os.path.isdir(model_path):
                config_file = os.path.join(model_path, "config.json")
                if os.path.exists(config_file):
                    file_size = os.path.getsize(config_file)
                    print(f"Classifier config.json size: {file_size} bytes")
            elif os.path.isfile(model_path):
                file_size = os.path.getsize(model_path)
                print(f"Classifier model file size: {file_size} bytes")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if the fine-tuned model exists, else fallback to base for development
        if os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Diagnostic: Verify if model.safetensors is binary or LFS pointer
            safetensors_file = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_file):
                size = os.path.getsize(safetensors_file)
                print(f"DEBUG: model.safetensors size: {size} bytes")
                if size < 500:
                    with open(safetensors_file, 'r') as f:
                        print(f"DEBUG: model.safetensors content: {f.read(100)}")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            print(f"Warning: Fine-tuned model not found at {model_path}. Loading base model.")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)
            
        self.model.to(self.device)
        
        # CPU Optimization: 8-bit Dynamic Quantization
        if self.device.type == 'cpu':
            print("Applying 8-bit quantization for CPU optimization...")
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            
        self.model.eval()
        
        self.classes = ['Secure', 'Misconfigured', 'SecretLeakage', 'HighRisk']

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        
        # Enable MC Dropout for uncertainty
        mean_probs, variance, predictive_entropy = get_mc_dropout_prediction(self.model, inputs)
        
        probs = mean_probs[0]
        pred_idx = np.argmax(probs)
        uncertainty = float(predictive_entropy[0])
        
        return {
            "prediction": self.classes[pred_idx],
            "confidence": float(probs[pred_idx]),
            "uncertainty": round(uncertainty, 4),
            "probabilities": {
                "secure": float(probs[0]),
                "misconfigured": float(probs[1]),
                "secretLeakage": float(probs[2]),
                "highRisk": float(probs[3])
            }
        }

import os

