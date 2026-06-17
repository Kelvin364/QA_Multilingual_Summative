"""
train_generator.py — fine-tune an open seq2seq model (mT5 / AfriTeVa) on the weak
Ghana configs (Aka_Gha, Eng_Gha) where retrieval is hard-capped (~99% unique answers).

Rubric-grade fine-tuning:
  - train + validation split with per-epoch eval loss  -> LEARNING CURVES (history.csv)
  - optional LoRA / PEFT  (--use-lora)                  -> PEFT experiment
  - swap model with --model-name                        -> model-selection experiment
  - decoding controls (--beams, --gen-len)              -> inference-settings experiment
  - seeded, checkpointed, version-robust                -> reproducibility

Self-contained (no project imports) so it runs standalone on Colab GPU. Open-licensed deps:
transformers, datasets, accelerate, sentencepiece, rouge-score, peft (Apache/MIT).

Outputs (in --output-dir/<run-name>):
  history.csv   : step/epoch train_loss & eval_loss  (for learning-curve plots)
  val_gen.csv   : ID, subset, input, reference, gen_pred   (for local Val-gating)
  test_gen.csv  : ID, subset, input, gen_pred              (to merge into submission)
  metrics.json  : per-subset generation mean-ROUGE vs the retrieval baseline + config
  model/        : saved model (full) or LoRA adapter

Usage (Colab):
  python train_generator.py --data-dir data --output-dir out --run-name mt5_full \
      --model-name google/mt5-base --subsets Aka_Gha,Eng_Gha --epochs 4
  python train_generator.py ... --run-name mt5_lora --use-lora     # PEFT variant
"""
from __future__ import annotations
import argparse, inspect, json, os, random
import numpy as np
import pandas as pd

# retrieval baseline to beat (our rerank+hybrid, Val mean-ROUGE), for the verdict print
RETRIEVAL_BASELINE = {"Aka_Gha": 0.356, "Eng_Gha": 0.332, "Amh_Eth": 0.031}


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def mean_rouge(refs, preds):
    """Our local metric: (ROUGE-1 F1 + ROUGE-L F1) / 2."""
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1 = rl = 0.0
    for r, p in zip(refs, preds):
        s = sc.score(str(r), str(p)); r1 += s["rouge1"].fmeasure; rl += s["rougeL"].fmeasure
    n = max(len(refs), 1)
    return 0.5 * (r1 / n + rl / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="out_generator")
    ap.add_argument("--run-name", default="mt5_full")
    ap.add_argument("--model-name", default="google/mt5-base")
    ap.add_argument("--subsets", default="Aka_Gha,Eng_Gha")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--train-bs", type=int, default=4)
    ap.add_argument("--eval-bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-input", type=int, default=128)
    ap.add_argument("--max-target", type=int, default=256)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out slice of TRAIN for eval-loss curve")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap train+val+test rows (0=all)")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (saves memory, ~2x slower; needed for full mT5-base on a T4)")
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    # Colab's default env may lack rouge-score; ensure it so scoring never crashes the run
    try:
        import rouge_score  # noqa: F401
    except ImportError:
        import subprocess, sys
        print("[setup] installing rouge-score ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rouge-score"], check=False)

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                              DataCollatorForSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)

    subsets = [s.strip() for s in args.subsets.split(",")]
    out_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    tr = pd.read_csv(os.path.join(args.data_dir, "Train.csv"))
    va = pd.read_csv(os.path.join(args.data_dir, "Val.csv"))
    te = pd.read_csv(os.path.join(args.data_dir, "Test.csv"))

    trL = tr[tr.subset.isin(subsets)].reset_index(drop=True)
    vaL = va[va.subset.isin(subsets)].reset_index(drop=True)
    teL = te[te.subset.isin(subsets)].reset_index(drop=True)
    if args.limit:   # debug smoke-test: cap rows
        trL, vaL, teL = trL.head(args.limit), vaL.head(args.limit), teL.head(args.limit)

    # held-out slice of TRAIN for the validation-loss learning curve (kept separate from
    # the official Val set, which we reserve for the generation-quality verdict)
    trL = trL.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_eval = max(1, int(len(trL) * args.val_frac))
    eval_split, train_split = trL.iloc[:n_eval], trL.iloc[n_eval:]
    print(f"train={len(train_split)} eval(curve)={len(eval_split)} val(verdict)={len(vaL)} "
          f"test={len(teL)} subsets={subsets} run={args.run_name} lora={args.use_lora}")

    tok = AutoTokenizer.from_pretrained(args.model_name)

    def to_input(df):  # language-country tag so one model serves both configs
        return [f"{s} | {q}" for s, q in zip(df.subset, df.input.astype(str))]

    def tokenize(df):
        mi = tok(to_input(df), max_length=args.max_input, truncation=True)
        lab = tok(text_target=df.output.astype(str).tolist(), max_length=args.max_target, truncation=True)
        return Dataset.from_dict({"input_ids": mi["input_ids"],
                                 "attention_mask": mi["attention_mask"],
                                 "labels": lab["input_ids"]})

    ds_train, ds_eval = tokenize(train_split), tokenize(eval_split)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lcfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=args.lora_r,
                          lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          target_modules=["q", "v"])   # T5/mT5 attention projections
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
    model.config.use_cache = False
    if args.grad_ckpt:                                # optional: saves memory, ~2x slower
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()        # needed for grad-checkpoint + PEFT
    collator = DataCollatorForSeq2Seq(tok, model=model)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # version-robust: eval_strategy (>=4.46) vs evaluation_strategy (<4.46)
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    strat_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    targ_kwargs = dict(
        output_dir=os.path.join(out_dir, "ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, warmup_ratio=0.05, weight_decay=0.01,
        logging_steps=25, save_strategy="no",   # no mid-train checkpoints (mT5 safetensors crash + faster); we save the final model below
        bf16=use_bf16, fp16=False, seed=args.seed, report_to=[],
    )
    targ_kwargs[strat_key] = "epoch"
    targs = Seq2SeqTrainingArguments(**targ_kwargs)

    try:
        trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=ds_train,
                                eval_dataset=ds_eval, data_collator=collator, processing_class=tok)
    except TypeError:
        trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=ds_train,
                                eval_dataset=ds_eval, data_collator=collator, tokenizer=tok)
    trainer.train()

    # learning-curve data: train_loss (per logging_steps) + eval_loss (per epoch)
    hist = pd.DataFrame(trainer.state.log_history)
    hist.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # ---- generation ----
    model.config.use_cache = True
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

    def generate(df, bs=16):
        outs, inputs = [], to_input(df)
        for i in range(0, len(inputs), bs):
            batch = tok(inputs[i:i + bs], return_tensors="pt", padding=True,
                       truncation=True, max_length=args.max_input).to(device)
            with torch.no_grad():
                gen = model.generate(**batch, max_length=args.gen_len, num_beams=args.beams,
                                     no_repeat_ngram_size=3)
            outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return outs

    # --- generate + SAVE FIRST, so a scoring/package error never loses the run ---
    vaL = vaL.copy(); vaL["gen_pred"] = generate(vaL)
    vaL[["ID", "subset", "input", "output", "gen_pred"]].rename(
        columns={"output": "reference"}).to_csv(os.path.join(out_dir, "val_gen.csv"), index=False)
    teL = teL.copy(); teL["gen_pred"] = generate(teL)
    teL[["ID", "subset", "input", "gen_pred"]].to_csv(os.path.join(out_dir, "test_gen.csv"), index=False)

    # save model: mT5 weights are non-contiguous views -> safetensors refuses them;
    # make contiguous and save as pytorch_model.bin (safe_serialization=False).
    for p in model.parameters():
        if not p.data.is_contiguous():
            p.data = p.data.contiguous()
    try:
        model.save_pretrained(os.path.join(out_dir, "model"), safe_serialization=False)
    except TypeError:
        model.save_pretrained(os.path.join(out_dir, "model"))
    tok.save_pretrained(os.path.join(out_dir, "model"))
    print(f"Saved val_gen.csv, test_gen.csv, model/ to {out_dir}")

    # --- score (best-effort; outputs already saved above) ---
    metrics = {}
    try:
        for s in subsets:
            m = vaL.subset == s
            gen_score = mean_rouge(vaL.loc[m, "output"].values, vaL.loc[m, "gen_pred"].values)
            base = RETRIEVAL_BASELINE.get(s)
            verdict = ("BEATS" if base and gen_score > base else "below") + (f" {base}" if base else "")
            metrics[s] = {"gen_mean_rouge": round(float(gen_score), 4), "retrieval_baseline": base, "n": int(m.sum())}
            print(f"[val] {s}: generation mean_rouge={gen_score:.4f}  ({verdict})  n={int(m.sum())}")
    except Exception as e:
        print(f"[warn] scoring skipped ({e}); val_gen.csv is saved — score at home with make_submission_v5.py")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"run_name": args.run_name, "model": args.model_name, "use_lora": args.use_lora,
                  "subsets": subsets, "epochs": args.epochs, "lr": args.lr, "beams": args.beams,
                  "per_subset": metrics}, f, indent=2)
    print(f"Saved history.csv, metrics.json to {out_dir}")


if __name__ == "__main__":
    main()
