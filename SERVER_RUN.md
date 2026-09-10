# Server training run — combined stories + conversation

Everything below the training step is already done locally. On the server you
build once and run one command.

## What's already prepared (local, in `data/`)

| file | size | what it is |
|---|---|---|
| `tok.bin` | 126 KB | BPE tokenizer, vocab 16004, already has `<|user|> <|bot|> <|endoftext|>` |
| `ts.tokens` | 920 MB | TinyStories, pre-tokenized (460.3M tokens) |
| `dialogues.txt` | 304 MB | 422k conversations: DailyDialog + 400k SODA + 9k Persona-Chat, in `__eou__` format |
| `dlg.tokens` | 148 MB | `dialogues.txt` pre-tokenized (74.0M tokens) |

Combined corpus = 534M tokens. Mixed at `dlgw=0.35` → ~1/3 of every batch is conversation.

## Copy to the server (keep the `data/` layout)

**Copy exactly these 4 files from `data/` (+ the source tree):**

| file | size | why |
|---|---|---|
| `data/tok.bin` | 126 KB | tokenizer — everything depends on it |
| `data/ts.tokens` | 920 MB | pre-tokenized stories (skips ~20 min re-tokenize) |
| `data/dlg.tokens` | 148 MB | pre-tokenized dialogue |
| `data/dialogues.txt` | 304 MB | **required even though dlg.tokens exists** — `gpt_train` only turns on dialogue mixing if this file opens |

Plus the source tree: `src/ include/ scripts/ Makefile`.
Total ≈ **1.37 GB**.

**Do NOT copy:**
- `TinyStories-train.txt` (1.9 GB) — not needed, `ts.tokens` is read directly
- `ckpt.bin` / `model.bin` — the old 12M model; you're training a new 25M architecture from scratch
- `Unconfirmed 480432.crdownload` — junk partial download
- `cifar-*`, `emnist-*`, `mnist`, `*-idx3-ubyte`, `dailydialog_extra/` — unrelated / already merged

## Commands, in order

### On the laptop (the dialogue corpus is already built — only redo if you change it)
```
# 1. (already done) build data/dialogues.txt + data/dlg.tokens
cd ~/Desktop/project/scratch
~/dlgenv/bin/python scripts/build_dialogue.py      # only if you want to change the mix
```

### Transfer to the server
```
# 2. copy source + the 4 data files (~1.37 GB)
rsync -avP ~/Desktop/project/scratch/src ~/Desktop/project/scratch/include \
  ~/Desktop/project/scratch/scripts ~/Desktop/project/scratch/Makefile \
  ~/Desktop/project/scratch/data/tok.bin ~/Desktop/project/scratch/data/ts.tokens \
  ~/Desktop/project/scratch/data/dlg.tokens ~/Desktop/project/scratch/data/dialogues.txt \
  USER@SERVER:~/gpt/
```

### On the server
```
# 3. build (sm_120 = RTX 50-series Blackwell; needs CUDA 12.8+)
cd ~/gpt
mkdir -p data && ls data          # tok.bin ts.tokens dlg.tokens dialogues.txt should be here
make gpt_train GPU_ARCH=sm_120

# 4. train  (~6-8 h at batch 24; TF32 is already on in context.h)
./gpt_train \
  dmodel=384 heads=6 layers=6 dff=1536 max_seq=1024 \
  block=512 batch=24 \
  steps=150000 warmup=2000 lr=6e-4 \
  text=data/TinyStories-train.txt text_cache=data/ts.tokens \
  dialogue=data/dialogues.txt   dialogue_cache=data/dlg.tokens   dlgw=0.35 \
  ckpt=data/ckpt.bin weights=data/model.bin \
  ckpt_every=2000 sample_every=2000

# 5. if it stops / disconnects — SAME command again, it resumes from data/ckpt.bin
#    (run it under tmux/screen/nohup so an SSH drop doesn't kill it)

# 6. chat with the result
make gpt_chat GPU_ARCH=sm_120
./gpt_chat data/tok.bin data/model.bin 0.8 40      # tok, weights, temperature, top_k
```

- **~25M params.** Fits easily in 16 GB at `batch=24`; raise to `batch=48` for ~20-30% faster, drop to `batch=12` if it OOMs.
- To restart training from scratch: `rm data/ckpt.bin data/model.bin` before step 4.
- `text=` path is passed but never opened (the `ts.tokens` cache is used); it just needs to be a valid string.

## What to expect

| loss | quality |
|---|---|
| 3.5+ | noise (first ~2k steps) |
| ~2.5 | simple phrases, broken |
| ~2.0 | mostly-grammatical sentences and short stories |
| ~1.6–1.8 | coherent stories + simple back-and-forth in the `[sample]` line |

150k steps ≈ 2.6 passes over the stories, ~9 over the dialogue mix. Watch the
`[sample] bot:` line printed every 2000 steps — that's the real signal for chat
ability, not the loss number.

## Chatting with the result

```
make gpt_chat
./gpt_chat data/tok.bin data/model.bin 0.8 40      # tok, weights, temperature, top_k
```

## Rebuilding the dialogue corpus (optional, if you want more/less)

```
~/dlgenv/bin/python scripts/build_dialogue.py     # edit SODA_MAX at top of file
rm data/dlg.tokens                                # force re-tokenize on next run
```
`~/dlgenv` is a venv with `datasets`+`pyarrow`. The SODA parquet (688 MB) is cached
in `~/.cache/huggingface/`.
