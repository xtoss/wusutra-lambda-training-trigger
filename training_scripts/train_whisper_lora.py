#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, traceback, glob, re, tarfile, tempfile, subprocess
from datetime import datetime
import argparse

print("🚀 start train_whisper_lora.py", flush=True)
print("argv:", sys.argv, flush=True)
print("cwd:", os.getcwd(), flush=True)

# Install dependencies if in SageMaker environment
def in_sagemaker():
    return os.path.exists("/opt/ml/input")

if in_sagemaker():
    print("📦 Installing dependencies in SageMaker environment...", flush=True)
    try:
        # Install from requirements.txt if available
        req_path = "/opt/ml/input/data/code/requirements.txt"
        if os.path.exists(req_path):
            print(f"📋 Installing from {req_path}", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
        else:
            # Install essential packages directly
            packages = [
                "librosa==0.10.1",
                "soundfile==0.12.1", 
                "transformers==4.30.2",
                "datasets==2.14.0",
                "accelerate==0.20.3",
                "peft==0.4.0",
                "huggingface_hub==0.16.4"
            ]
            for pkg in packages:
                print(f"📦 Installing {pkg}", flush=True)
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        print("✅ Dependencies installed successfully", flush=True)
    except Exception as e:
        print(f"⚠️ Error installing dependencies: {e}", flush=True)
        # Continue anyway - some packages might already be available

def safe_listdir(p):
    try:
        return os.listdir(p)
    except Exception as e:
        print("listdir error", p, e, flush=True)
        return []

def find_hf_dir(root):
    for r, d, f in os.walk(root):
        if "config.json" in f:
            return r
    return root

def ensure_local_base_model(base_root):
    if not os.path.exists(base_root):
        raise FileNotFoundError(f"base model path not found {base_root}")
    tar_path = os.path.join(base_root, "model.tar.gz")
    if os.path.isfile(tar_path):
        tmp = tempfile.mkdtemp(prefix="extracted_model_")
        print("extracting", tar_path, "to", tmp, flush=True)
        with tarfile.open(tar_path, "r:gz") as t:
            t.extractall(tmp)
        return find_hf_dir(tmp)
    return find_hf_dir(base_root)

# ----------------- argparse -----------------
p = argparse.ArgumentParser()
p.add_argument("--audio_path", type=str, default="/opt/ml/input/data/training")
p.add_argument("--model_output_path", type=str, default="/opt/ml/model")
p.add_argument("--num_train_epochs", type=int, default=10)
p.add_argument("--training_type", choices=["full", "incremental", "lora"], default="lora")
p.add_argument("--base_model", default=None)
p.add_argument("--min_timestamp", default=None)
p.add_argument("--validation_path", type=str, default="/opt/ml/input/data/validation")
p.add_argument("--run_validation", type=bool, default=True)
p.add_argument("--push_to_hub", type=bool, default=False)
p.add_argument("--hub_model_id", type=str, default="jxue/whisper_small_jiangyin_lora")
p.add_argument("--hub_token", type=str, default=None)
# LoRA specific arguments
p.add_argument("--lora_r", type=int, default=16)
p.add_argument("--lora_alpha", type=int, default=32)
p.add_argument("--lora_dropout", type=float, default=0.1)
args, unknown = p.parse_known_args()
print("unknown args:", unknown, flush=True)

if in_sagemaker():
    for d in [
        "/opt/ml/input",
        "/opt/ml/input/data",
        "/opt/ml/input/data/training",
        "/opt/ml/input/data/validation",
        "/opt/ml/input/data/base",
        "/opt/ml/model",
    ]:
        print(d, "exists" if os.path.exists(d) else "missing", safe_listdir(d)[:5], flush=True)

# ----------------- imports -----------------
try:
    import numpy as np
    import librosa
    import json
    import torch
    from datasets import Dataset
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, TrainingArguments, Trainer
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union, Optional
    if args.training_type == "lora":
        from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except Exception as e:
    print("❌ import error", e, flush=True)
    traceback.print_exc()
    sys.exit(2)

# ----------------- model load -----------------
try:
    # Determine base model
    if args.training_type == "incremental" or args.training_type == "lora":
        local_base = None
        if os.path.isdir("/opt/ml/input/data/base"):
            try:
                local_base = ensure_local_base_model("/opt/ml/input/data/base")
            except Exception as e:
                print("base channel not usable", e, flush=True)
        model_loaded = False
        if not local_base and args.base_model:
            if args.base_model.startswith("s3://"):
                raise ValueError("base_model is s3 uri, use TrainingInput(channel='base') to mount it")
            # Check if it's a Hugging Face model ID (contains slash) or local path
            if "/" in args.base_model and not os.path.exists(args.base_model):
                # It's a Hugging Face model ID like "openai/whisper-small"
                print(f"🤗 Using Hugging Face model: {args.base_model}", flush=True)
                model_id = args.base_model
                processor = WhisperProcessor.from_pretrained(model_id)
                model = WhisperForConditionalGeneration.from_pretrained(model_id)
                model_loaded = True
            else:
                # It's a local path
                local_base = ensure_local_base_model(args.base_model)
        
        if not model_loaded:
            if not local_base:
                print("⚠️ fallback to openai/whisper-small", flush=True)
                model_id = "openai/whisper-small"
                processor = WhisperProcessor.from_pretrained(model_id)
                model = WhisperForConditionalGeneration.from_pretrained(model_id)
            else:
                print("📥 loading base from", local_base, flush=True)
                processor = WhisperProcessor.from_pretrained(local_base)
                model = WhisperForConditionalGeneration.from_pretrained(local_base)
    else:
        model_id = "openai/whisper-small"
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    # Apply LoRA if specified
    if args.training_type == "lora":
        print("🎯 Applying LoRA configuration...", flush=True)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"]
        )
        model = get_peft_model(model, lora_config)
        print("✅ LoRA applied successfully", flush=True)
        model.print_trainable_parameters()
        
        # Ensure model is in training mode
        model.train()
    else:
        # Original freezing logic for full/incremental
        print("🔒 Freezing encoder layers...", flush=True)
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        
        # Print trainable parameter stats
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"📊 Model parameters:", flush=True)
        print(f"   Total: {total_params:,}", flush=True)
        print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)", flush=True)
        print(f"   Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)", flush=True)
    
    print("✅ model ready", flush=True)
except Exception as e:
    print("❌ model load failed", e, flush=True)
    traceback.print_exc()
    sys.exit(2)

# ----------------- data load -----------------
pat = re.compile(r"^(\d{8}-\d{6})-([^-]+)-(.+?)(?:--(.+?))?\.wav$", re.IGNORECASE)
audio_glob = glob.glob(os.path.join(args.audio_path, "**", "*.wav"), recursive=True)
print("📂 scan", args.audio_path, "found", len(audio_glob), "wav", flush=True)

raw = []
min_ts = None
if args.training_type == "incremental" and args.min_timestamp:
    try:
        min_ts = int(args.min_timestamp.replace("-", ""))
        print(f"📅 Using min_timestamp {args.min_timestamp} for incremental training", flush=True)
    except Exception as e:
        print(f"⚠️ Error parsing min_timestamp {args.min_timestamp}: {e}", flush=True)

for ap in sorted(audio_glob):
    fn = os.path.basename(ap)
    m = pat.match(fn)
    if not m:
        print("skip invalid", fn, flush=True)
        continue
    if min_ts:
        ts = int(m.group(1).replace("-", ""))
        if ts <= min_ts:
            continue
    try:
        y, sr = librosa.load(ap, sr=16000)
        txt = re.sub(r"-", "", m.group(3))
        raw.append({"audio": {"array": y, "sampling_rate": sr}, "text": txt})
    except Exception as e:
        print("⚠️ load error", fn, e, flush=True)

if not raw:
    print("❌ no training data after filtering", flush=True)
    sys.exit(2)

ds = Dataset.from_list(raw)

def preprocess(ex):
    # Use the same approach as the working train_whisper.py
    a = ex["audio"]
    it = processor(a["array"], sampling_rate=a["sampling_rate"], return_tensors="np")
    ex["input_features"] = it["input_features"][0]  # Remove batch dimension like train_whisper.py
    ex["labels"] = processor.tokenizer(ex["text"]).input_ids
    return ex

# Process the dataset - keep only the columns we need
original_columns = ds.column_names
print(f"🔍 Original dataset columns: {original_columns}", flush=True)

# Map the preprocessing function
print(f"🔄 About to preprocess {len(ds)} examples", flush=True)
print(f"🔍 Sample raw data before preprocessing: {ds[0] if len(ds) > 0 else 'No data'}", flush=True)

ds = ds.map(preprocess, remove_columns=ds.column_names)

print(f"🔄 Finished preprocessing. Dataset now has {len(ds)} examples", flush=True)

# Ensure the dataset has the correct columns
if "input_features" not in ds.column_names or "labels" not in ds.column_names:
    print(f"❌ Dataset missing required columns. Has: {ds.column_names}", flush=True)
    raise ValueError(f"Dataset must have 'input_features' and 'labels' columns, but has: {ds.column_names}")

# Set format exactly like the working train_whisper.py
ds.set_format(type="torch", columns=["input_features", "labels"])

print(f"🔍 Dataset columns after processing: {ds.column_names}", flush=True)
print(f"🔍 Dataset format: {ds.format}", flush=True)
if len(ds) > 0:
    sample = ds[0]
    print(f"🔍 First item keys: {list(sample.keys())}", flush=True)
    print(f"🔍 First item type: {type(sample)}", flush=True)
    print(f"🔍 First item input_features shape: {sample['input_features'].shape if 'input_features' in sample else 'Not found'}", flush=True)
    print(f"🔍 First item labels shape: {sample['labels'].shape if 'labels' in sample else 'Not found'}", flush=True)
    
    # Test a small batch to see if the issue occurs during batching
    batch = ds[:2]
    print(f"🔍 Batch keys: {list(batch.keys())}", flush=True)
    if 'input_features' in batch:
        print(f"🔍 Batch input_features shape: {batch['input_features'].shape}", flush=True)
    if 'labels' in batch:
        print(f"🔍 Batch labels shape: {batch['labels'].shape}", flush=True)

# ----------------- training -----------------
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
# Determine output directory based on training type
if args.training_type == "lora":
    dir_prefix = "whisper_lora_output"
elif args.training_type == "incremental":
    dir_prefix = "whisper_incremental_output"
else:
    dir_prefix = "whisper_finetuned_output"

out_dir = (
    args.model_output_path
    if args.model_output_path == "/opt/ml/model"
    else os.path.join(args.model_output_path, f"{dir_prefix}_{ts}")
)
# Learning rate based on training type
if args.training_type == "lora":
    lr = 3e-4  # Higher LR for LoRA
elif args.training_type == "incremental":
    lr = 1e-6
else:
    lr = 5e-6

print("📝 output_dir", out_dir, "lr", lr, "training_type", args.training_type, flush=True)

ta = TrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=1,
    num_train_epochs=args.num_train_epochs,
    evaluation_strategy="no",
    logging_steps=10,
    save_strategy="no",
    save_total_limit=1,
    remove_unused_columns=False,  # Same as train_whisper.py
    report_to=[],
    learning_rate=lr,
    fp16=True,  # Use FP16 on GPU for efficiency
)

# Custom data collator for Whisper
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    padding: Union[bool, str] = True
    return_tensors: str = "pt"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Debug: Print feature keys to understand the structure
        if features:
            print(f"🔍 Data collator received {len(features)} features", flush=True)
            print(f"🔍 First feature keys: {list(features[0].keys())}", flush=True)
            print(f"🔍 Feature type: {type(features[0])}", flush=True)
            if "input_features" in features[0]:
                print(f"🔍 input_features shape: {features[0]['input_features'].shape if hasattr(features[0]['input_features'], 'shape') else type(features[0]['input_features'])}", flush=True)
            if "labels" in features[0]:
                print(f"🔍 labels shape: {features[0]['labels'].shape if hasattr(features[0]['labels'], 'shape') else type(features[0]['labels'])}", flush=True)
        
        # Ensure we have the expected structure
        if not features:
            raise ValueError("No features provided to data collator")
        
        # Check if features are properly structured
        for i, feature in enumerate(features):
            if not isinstance(feature, dict):
                print(f"❌ Feature {i} is not a dict: {type(feature)}", flush=True)
                print(f"❌ Feature content: {feature}", flush=True)
                raise TypeError(f"Feature {i} must be a dictionary, got {type(feature)}")
            
            # Check for required keys
            if "input_features" not in feature:
                print(f"❌ KeyError in data collator: 'input_features'", flush=True)
                print(f"❌ Available keys in feature: {list(feature.keys())}", flush=True)
                print(f"❌ Feature type: {type(feature)}", flush=True)
                if feature:
                    print(f"❌ First feature: {feature}", flush=True)
                raise KeyError(f"'input_features' key missing from feature {i}")
                
            if "labels" not in feature:
                print(f"❌ KeyError in data collator: 'labels'", flush=True)  
                print(f"❌ Available keys in feature: {list(feature.keys())}", flush=True)
                print(f"❌ Feature type: {type(feature)}", flush=True)
                if feature:
                    print(f"❌ First feature: {feature}", flush=True)
                raise KeyError(f"'labels' key missing from feature {i}")
        
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        try:
            # Convert to torch tensors if needed and extract features
            input_features = []
            label_features = []
            
            for i, feature in enumerate(features):
                # Handle input_features
                inp_feat = feature["input_features"]
                if not isinstance(inp_feat, torch.Tensor):
                    if hasattr(inp_feat, '__array__'):
                        inp_feat = torch.from_numpy(inp_feat)
                    else:
                        inp_feat = torch.tensor(inp_feat)
                
                # Handle labels
                labels = feature["labels"]
                if not isinstance(labels, torch.Tensor):
                    if hasattr(labels, '__array__'):
                        labels = torch.from_numpy(labels)
                    else:
                        labels = torch.tensor(labels)
                
                input_features.append({"input_features": inp_feat})
                label_features.append({"input_ids": labels})
            
            print(f"✅ Successfully extracted {len(input_features)} input features and {len(label_features)} label features", flush=True)
            
        except KeyError as e:
            print(f"❌ KeyError in data collator: {e}", flush=True)
            print(f"Available keys in feature: {list(features[0].keys()) if features else 'No features'}", flush=True)
            print(f"Feature type: {type(features[0]) if features else 'No features'}", flush=True)
            if features:
                print(f"First feature: {features[0]}", flush=True)
            raise
        except Exception as e:
            print(f"❌ Error extracting features: {e}", flush=True)
            print(f"Features type: {type(features)}", flush=True)
            if features:
                print(f"First feature content: {features[0]}", flush=True)
            raise

        batch = self.processor.feature_extractor.pad(input_features, return_tensors=self.return_tensors)

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors=self.return_tensors)

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Custom trainer to handle Whisper's special input format
class WhisperTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract the proper inputs for Whisper
        input_features = inputs.get("input_features")
        labels = inputs.get("labels")
        
        # Forward pass - for PEFT models we need to call the base model directly
        if hasattr(model, 'base_model'):
            outputs = model.base_model.model(input_features=input_features, labels=labels)
        else:
            outputs = model(input_features=input_features, labels=labels)
        
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

trainer = WhisperTrainer(
    model=model, 
    args=ta, 
    train_dataset=ds
    # Remove custom data_collator - use default like train_whisper.py
)

print("🚀 start training...", flush=True)
print(f"🔍 Final dataset info: {len(ds)} examples, columns: {ds.column_names}", flush=True)
if len(ds) > 0:
    try:
        sample = ds[0]
        print(f"🔍 Final sample keys: {list(sample.keys())}", flush=True)
        print(f"🔍 Final sample types: {[(k, type(v)) for k, v in sample.items()]}", flush=True)
    except Exception as e:
        print(f"❌ Error accessing sample: {e}", flush=True)

trainer.train()

print("💾 saving...", flush=True)
if args.training_type == "lora":
    # Save LoRA adapter with safetensors format
    model.save_pretrained(out_dir, safe_serialization=True)
    processor.save_pretrained(out_dir)
else:
    # Save full model
    model.save_pretrained(out_dir, safe_serialization=True)
    processor.save_pretrained(out_dir)

# ----------------- HuggingFace Hub upload for LoRA -----------------
# Auto-push if HF token is available (from args or environment)
hf_token = args.hub_token or os.environ.get('HF_TOKEN')
if args.training_type == "lora" and (args.push_to_hub or hf_token):
    print("🤗 Uploading LoRA model to HuggingFace Hub...", flush=True)
    try:
        # Use token from args or environment
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        
        # Push to existing hub repo (don't create new repo)
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Check if repo exists before pushing
        try:
            api.repo_info(args.hub_model_id, token=hf_token)
            print(f"✅ Repository {args.hub_model_id} found, uploading to existing repo...", flush=True)
        except Exception as e:
            print(f"❌ Repository {args.hub_model_id} not found or not accessible: {e}", flush=True)
            raise
        
        # Push to hub without trying to create repo (skip README.md)
        model.push_to_hub(
            args.hub_model_id,
            commit_message=f"Upload LoRA adapter trained on {len(ds)} samples",
            token=hf_token,
            create_pr=False,
            safe_serialization=True,
            ignore_patterns=["README.md", "*.md"]  # Keep existing README on HF
        )
        processor.push_to_hub(
            args.hub_model_id,
            token=hf_token,
            create_pr=False,
            ignore_patterns=["README.md", "*.md"]  # Keep existing README on HF
        )
        
        print(f"✅ Successfully uploaded to https://huggingface.co/{args.hub_model_id}", flush=True)
    except Exception as e:
        print(f"⚠️ Failed to upload to HuggingFace Hub: {e}", flush=True)
        traceback.print_exc()

# ----------------- validation -----------------
if args.run_validation and os.path.exists(args.validation_path):
    print("\n🔍 Starting validation...", flush=True)
    
    # CER calculation function
    def cer(ref: str, hyp: str) -> float:
        r, h = list(ref), list(hyp)
        n, m = len(r), len(h)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if r[i - 1] == h[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,        # delete
                    dp[i][j - 1] + 1,        # insert
                    dp[i - 1][j - 1] + cost  # substitute
                )
        return dp[n][m] / max(1, n)
    
    # Load validation data
    val_glob = glob.glob(os.path.join(args.validation_path, "**", "*.wav"), recursive=True)
    print(f"📂 Found {len(val_glob)} validation files", flush=True)
    
    if val_glob:
        # Prepare model for inference
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        # Force Chinese transcription
        forced_ids = None
        try:
            forced_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
        except:
            pass
        
        total_cer = 0
        valid_samples = 0
        
        for vp in val_glob[:50]:  # Limit to 50 samples for speed
            fn = os.path.basename(vp)
            m = pat.match(fn)
            if not m:
                continue
            
            try:
                # Load audio
                y, sr = librosa.load(vp, sr=16000)
                ref_text = re.sub(r"-", "", m.group(3))
                
                # Process audio
                inputs = processor(y, sampling_rate=sr, return_tensors="pt")
                input_features = inputs.input_features.to(device)
                
                # Generate prediction
                with torch.no_grad():
                    if forced_ids is not None:
                        # PEFT models require keyword arguments
                        predicted_ids = model.generate(input_features=input_features, forced_decoder_ids=forced_ids)
                    else:
                        # PEFT models require keyword arguments
                        predicted_ids = model.generate(input_features=input_features)
                
                # Decode
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
                hyp_text = transcription[0].strip() if transcription else ""
                
                # Calculate CER
                sample_cer = cer(ref_text, hyp_text)
                total_cer += sample_cer
                valid_samples += 1
                
                if valid_samples <= 100:  # Show first 100 examples
                    print(f"  REF: {ref_text}")
                    print(f"  HYP: {hyp_text}")
                    print(f"  CER: {sample_cer:.4f}")
                    print("  ---")
                
            except Exception as e:
                print(f"⚠️ Validation error for {fn}: {e}", flush=True)
        
        if valid_samples > 0:
            avg_cer = total_cer / valid_samples
            print(f"\n📊 Validation Results:")
            print(f"   Samples: {valid_samples}")
            print(f"   Average CER: {avg_cer:.4f}")
            print(f"   Average WER (approx): {avg_cer * 2:.4f}")  # Rough approximation
            
            # Save validation report
            val_report = {
                "model_path": out_dir,
                "timestamp": datetime.now().isoformat(),
                "training_type": args.training_type,
                "samples": valid_samples,
                "cer": round(avg_cer, 4),
                "wer_approx": round(avg_cer * 2, 4)
            }
            
            if args.training_type == "lora":
                val_report["lora_config"] = {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout
                }
            
            with open(os.path.join(out_dir, "validation_report.json"), "w") as f:
                json.dump(val_report, f, indent=2)
            
            # Update model output directory name with CER
            if args.model_output_path != "/opt/ml/model":
                # For local training, rename folder to include CER
                cer_pct = int(avg_cer * 100)
                new_out_dir = out_dir.replace(f"{dir_prefix}_{ts}", f"{dir_prefix}_{ts}_cer{cer_pct}")
                try:
                    os.rename(out_dir, new_out_dir)
                    print(f"📁 Model saved to: {new_out_dir}")
                    out_dir = new_out_dir
                except:
                    print(f"⚠️ Could not rename output dir, CER={cer_pct}%", flush=True)
        else:
            print("⚠️ No valid validation samples processed", flush=True)
    else:
        print("⚠️ No validation files found", flush=True)
else:
    print("ℹ️ Skipping validation (no validation path or disabled)", flush=True)

print("✅ done", out_dir, flush=True)