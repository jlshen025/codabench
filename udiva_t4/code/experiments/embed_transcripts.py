"""Embed each transcript utterance with cached bge-m3 (multilingual; Spanish).
Saves per-session npz: emb (n_utt x dim) fp16, starts, ends, spk (0=a,1=b,2=sup).
Run: python embed_transcripts.py   (all sessions; cuda if available)
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, glob, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from udiva import io as IO, feats as FT

OUT = FT.TEXT_DIR                        # $UDIVA_TEXT_DIR
MODEL = "BAAI/bge-m3"
SPK = {"participant_a": 0, "participant_b": 1, "supervisor": 2}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval().to(dev)
    for sess in IO.list_sessions():
        utts = IO.load_transcript(sess)
        texts = [u[3] for u in utts]
        starts = np.array([u[0] for u in utts], dtype=np.float32)
        ends = np.array([u[1] for u in utts], dtype=np.float32)
        spk = np.array([SPK.get(u[2], 2) for u in utts], dtype=np.int8)
        embs = []
        for b in range(0, len(texts), 32):
            batch = texts[b:b + 32]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**enc)
                # CLS embedding (bge uses [CLS]); L2-normalize
                emb = out.last_hidden_state[:, 0]
                emb = torch.nn.functional.normalize(emb, dim=1)
            embs.append(emb.float().cpu().numpy())
        E = np.concatenate(embs, 0).astype(np.float16) if embs else np.zeros((0, 1024), np.float16)
        np.savez_compressed(os.path.join(OUT, f"{sess}.npz"), emb=E, starts=starts, ends=ends, spk=spk)
        print(f"{sess}: {E.shape} dev={dev}", flush=True)
    od = os.environ.get("EXP_OUTPUT_DIR", ".")
    json.dump({"ok": True, "model": MODEL}, open(os.path.join(od, "result.json"), "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
