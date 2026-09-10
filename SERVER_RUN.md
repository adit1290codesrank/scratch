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

Minimum payload (~1.07 GB, skips all re-tokenizing):

```
scratch/                     # whole source tree: src/ include/ scripts/ Makefile
scratch/data/tok.bin
scratch/data/ts.tokens
scratch/data/dlg.tokens
scratch/data/dialogues.txt   # REQUIRED even though dlg.tokens exists —
                             # gpt_train checks this file exists to enable dialogue
```

You do NOT need `TinyStories-train.txt` on the server (the `.tokens` cache is used directly).

## Build + run

```
cd scratch
make gpt_train

./gpt_train \
  dmodel=384 heads=6 layers=6 dff=1536 max_seq=1024 \
  block=512 batch=24 \
  steps=150000 warmup=2000 lr=6e-4 \
  text=data/TinyStories-train.txt text_cache=data/ts.tokens \
  dialogue=data/dialogues.txt   dialogue_cache=data/dlg.tokens   dlgw=0.35 \
  ckpt=data/ckpt.bin weights=data/model.bin \
  ckpt_every=2000 sample_every=2000
```

- **~25M params.** Fits comfortably on any 12 GB+ GPU at `batch=24`. Raise `batch` to
  48–64 if the server GPU is bigger (faster); drop to 12–16 if it OOMs.
- **Resumable:** re-run the exact same command, it picks up `data/ckpt.bin`.
- **LR** warms up over 2000 steps to 6e-4, then cosine-decays to 3e-5 by step 150000.
- Start fresh: `rm data/ckpt.bin data/model.bin` first.

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
