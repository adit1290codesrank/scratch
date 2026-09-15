# Server training run — text-to-SQL IR prediction

This is a separate model from the conversational one (`SERVER_RUN.md`) — its
own tokenizer, its own checkpoint, its own training script. Nothing here
touches `data/tok.bin`, `data/ckpt.bin`, or `data/model.bin`.

## What's already prepared (local, in `data/`)

| file | size | what it is |
|---|---|---|
| `ir_pairs.txt` | 3.7 MB | 15,360 `question<TAB>ir_json` lines, generated + validated against the real compiler |
| `ir_corpus.txt` | 3.2 MB | same content, one item per line — tokenizer training input only |

## Why a fresh tokenizer, not the conversational one

Tested both against real schema identifiers before deciding — the general
16k-vocab tokenizer badly fragments exactly the strings that need to come out
byte-exact for the compiler to accept them:

| identifier | general (16k) | fresh (6k, trained on `ir_corpus.txt`) |
|---|---|---|
| `AN_LOGISTICS_TRACKER` | 17 tokens | 5 |
| `ANDashBoardVehicleUtilization` | 14 tokens | 1 |
| `TransporterAssignedDateTime` | 11 tokens | 1 |
| `CB_OBDMassUpdate` | 11 tokens | 3 |

~3x more tokens overall on a 13-string sample. Every extra token is another
chance to emit a name the compiler rejects. Train fresh — see step 1 below.

## Commands, in order

### Transfer to the server
```
rsync -avP ~/Desktop/project/scratch/src ~/Desktop/project/scratch/include \
  ~/Desktop/project/scratch/scripts ~/Desktop/project/scratch/Makefile \
  ~/Desktop/project/scratch/data/ir_pairs.txt ~/Desktop/project/scratch/data/ir_corpus.txt \
  USER@SERVER:~/gpt/
```

### On the server
```
cd ~/gpt

# 1. train the tokenizer (~1s, vocab=6000 already looked good locally —
#    try 8000 too and compare fragmentation before committing if you want)
make bpe_train
./bpe_train data/ir_tok.bin 6000 data/ir_corpus.txt

# 2. build
make gpt_train_ir GPU_ARCH=sm_120

# 3. train
./gpt_train_ir \
  tok=data/ir_tok.bin ckpt=data/ir_ckpt.bin weights=data/ir_model.bin \
  text=data/ir_pairs.txt text_cache=data/ir_pairs.cache \
  dmodel=384 heads=6 layers=6 dff=1536 max_seq=256 block=128 batch=48 \
  steps=20000 warmup=500 lr=4e-4 wd=0.01 dropout=0 \
  ckpt_every=500 sample_every=500

# 4. if it stops / disconnects — same command again, resumes from data/ir_ckpt.bin
#    (run under tmux/screen/nohup so an SSH drop doesn't kill it)
```

Needs a `gpt_train_ir:` target in the Makefile (copy the existing `gpt_train:`
target, point it at `scripts/gpt_train_ir.cpp`).

## What to expect

Watch the `[sample] ir:` lines printed every 500 steps — real signal is
whether the JSON is well-formed and references real tables/columns, not the
loss number in isolation. A quick sanity check worth running once you have a
checkpoint: pipe a `[sample]` line straight into `ir_batch_check` (from
`sql-compiler/`) and see if it actually validates.

## Restarting from scratch

`rm data/ir_ckpt.bin data/ir_model.bin` before step 3. Re-running step 1 with
a different vocab size requires this too, since the checkpoint embeds the
vocab size.
