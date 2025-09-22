#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, traceback, glob, re, tarfile, tempfile
from datetime import datetime
import argparse

print("🚀 start train_whisper.py", flush=True)
print("argv:", sys.argv, flush=True)
print("cwd:", os.getcwd(), flush=True)

def in_sagemaker():
    return os.path.exists("/opt/ml/input")

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
p.add_argument("--mode", choices=["full", "incremental"], default="full")
p.add_argument("--base_model", default=None)
p.add_argument("--min_timestamp", default=None)
p.add_argument("--validation_path", type=str, default="/opt/ml/input/data/validation")
p.add_argument("--run_validation", type=bool, default=True)
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
except Exception as e:
    print("❌ import error", e, flush=True)
    traceback.print_exc()
    sys.exit(2)

# ----------------- model load -----------------
try:
    if args.mode == "incremental":
        local_base = None
        if os.path.isdir("/opt/ml/input/data/base"):
            try:
                local_base = ensure_local_base_model("/opt/ml/input/data/base")
            except Exception as e:
                print("base channel not usable", e, flush=True)
        if not local_base and args.base_model:
            if args.base_model.startswith("s3://"):
                raise ValueError("base_model is s3 uri, use TrainingInput(channel='base') to mount it")
            local_base = ensure_local_base_model(args.base_model)
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
    
    # Freeze encoder layers
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
if args.mode == "incremental" and args.min_timestamp:
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
    a = ex["audio"]
    it = processor(a["array"], sampling_rate=a["sampling_rate"], return_tensors="np")
    ex["input_features"] = it["input_features"][0]
    ex["labels"] = processor.tokenizer(ex["text"]).input_ids
    return ex

ds = ds.map(preprocess, remove_columns=ds.column_names)
ds.set_format(type="torch", columns=["input_features", "labels"])

# ----------------- training -----------------
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
# Determine output directory based on mode
if args.mode == "incremental":
    dir_prefix = "whisper_incremental_output"
else:
    dir_prefix = "whisper_finetuned_output"

out_dir = (
    args.model_output_path
    if args.model_output_path == "/opt/ml/model"
    else os.path.join(args.model_output_path, f"{dir_prefix}_{ts}")
)
lr = 1e-6 if args.mode == "incremental" else 5e-6
print("📝 output_dir", out_dir, "lr", lr, flush=True)

ta = TrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=1,
    num_train_epochs=args.num_train_epochs,
    evaluation_strategy="no",
    logging_steps=10,
    save_strategy="no",
    save_total_limit=1,
    remove_unused_columns=False,
    report_to=[],
    learning_rate=lr,
)
trainer = Trainer(model=model, args=ta, train_dataset=ds)

print("🚀 start training...", flush=True)
trainer.train()

print("💾 saving...", flush=True)
model.save_pretrained(out_dir, safe_serialization=True)
processor.save_pretrained(out_dir)

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
                        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_ids)
                    else:
                        predicted_ids = model.generate(input_features)
                
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
                "samples": valid_samples,
                "cer": round(avg_cer, 4),
                "wer_approx": round(avg_cer * 2, 4)
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
