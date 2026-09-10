#!/usr/bin/env python3
"""
Build data/dialogues.txt in the format DataLoader::build_dialogue expects:
  - one conversation per line
  - turns joined by " __eou__ ", line ends with " __eou__"
  - turns alternate user/bot starting with user

Sources: existing DailyDialog + SODA + Synthetic-Persona-Chat (all from the HF cache).
Needs: ~/dlgenv with `datasets`/`pyarrow` and the two files already downloaded via
huggingface_hub. One-time job; copy data/dialogues.txt to the server afterwards.
"""
import os, re, csv, glob

OUT = "data/dialogues.txt"
SODA_MAX = 400_000            # cap SODA dialogues so the file stays a sane size

csv.field_size_limit(10 ** 7)

def clean(s):
    s = str(s).replace("\n", " ").replace("\r", " ").replace("__eou__", " ")
    s = s.replace("[user 1's name]", "Alex").replace("[user 2's name]", "Sam")
    s = re.sub(r"\[[^\]]{0,40}\]", "", s)      # drop leftover [placeholders]
    s = re.sub(r"\s+", " ", s).strip()
    return s

def emit(fh, turns):
    turns = [clean(t) for t in turns]
    turns = [t for t in turns if t]
    if len(turns) < 2:
        return 0
    fh.write(" __eou__ ".join(turns) + " __eou__\n")
    return 1

def cache_file(repo, needle):
    hits = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/datasets--{repo}/snapshots/*/{needle}"))
    return hits[0] if hits else None

os.makedirs("data", exist_ok=True)
n = 0
with open(OUT, "w", encoding="utf-8") as out:

    # 1) DailyDialog (already one-per-line, __eou__ separated)
    src = "data/dialogues_text.txt"
    if os.path.exists(src):
        with open(src, encoding="utf-8") as f:
            for line in f:
                n += emit(out, line.split("__eou__"))
        print(f"dailydialog: total {n}", flush=True)

    # 2) SODA - clean two-party social dialogues
    pq = cache_file("allenai--soda", "train.parquet")
    if pq:
        import pyarrow.parquet as pqmod
        pf = pqmod.ParquetFile(pq)
        got = 0
        for batch in pf.iter_batches(batch_size=20000, columns=["dialogue"]):
            for d in batch.column("dialogue"):
                if emit(out, [t.as_py() for t in d]):
                    got += 1; n += 1
                if got >= SODA_MAX:
                    break
            print(f"  soda {got}/{SODA_MAX}", flush=True)
            if got >= SODA_MAX:
                break
        print(f"soda: +{got}  total {n}", flush=True)
    else:
        print("soda: parquet not in cache, skipped", flush=True)

    # 3) Synthetic-Persona-Chat
    pc = cache_file("google--Synthetic-Persona-Chat", "data/Synthetic-Persona-Chat_train.csv")
    if pc:
        got = 0
        with open(pc, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                conv = row.get("Best Generated Conversation", "")
                turns = re.split(r"(?:^|\n)\s*(?:User|Person)\s*\d*\s*:\s*", conv)
                if emit(out, [t for t in turns if t.strip()]):
                    got += 1; n += 1
        print(f"persona-chat: +{got}  total {n}", flush=True)
    else:
        print("persona-chat: csv not in cache, skipped", flush=True)

sz = os.path.getsize(OUT) / 1e6
print(f"\nwrote {OUT}: {n} conversations, {sz:.1f} MB")
