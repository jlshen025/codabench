"""
predict_test.py — Deployable inference: saved encoder(s) + LLM → soft-vote → .txt submission.

At eval: (1) run llm_predict.py --order file on the test CSV → llm npz (proba aligned to file order);
(2) run this with the full-data encoder dirs + that npz. Produces predictions.txt (ONE label per
line, test ROW order) zipped FLAT — the confirmed Codabench format.

If the input CSV has a 'stance' column (e.g. dev.csv as a stand-in), prints Favg2 for validation.
"""
import os, sys, json, argparse, zipfile
os.environ.setdefault("HF_HOME", "<cache>/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S


@torch.no_grad()
def encoder_proba(model_dir, targets, texts, max_len, device, batch=64):
    tok = AutoTokenizer.from_pretrained(model_dir)
    is_mtl = os.path.exists(os.path.join(model_dir, "mtl_state.pt"))
    if is_mtl:
        from train_mtl import MTLModel
        cfg = json.load(open(os.path.join(model_dir, "mtl_config.json")))
        model = MTLModel(cfg["base_model"]).to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, "mtl_state.pt"), map_location=device, weights_only=True))
        model.eval()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(targets[i:i+batch], texts[i:i+batch], truncation=True, padding=True,
                  max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(**enc)[0] if is_mtl else model(**enc).logits
        out.append(torch.softmax(logits.float(), 1).cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="saved encoder dirs (base arch)")
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--preprocess", default="baseline")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--llm_npz", default="", help="precomputed LLM proba (order=file, aligned to test_csv)")
    ap.add_argument("--w_llm", type=float, default=0.4)
    ap.add_argument("--out_zip", required=True)
    ap.add_argument("--txt_name", default="predictions.txt")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.test_csv, keep_default_na=False, dtype=str)
    for c in ("text", "target"):
        df[c] = df[c].astype(str).str.strip()
    text_proc = df["text"].apply(S.PREPROCESSORS[args.preprocess]).tolist()
    targets = df["target"].tolist()

    enc = np.mean([encoder_proba(m, targets, text_proc, args.max_len, device) for m in args.models], axis=0)
    print(f"[enc] {len(args.models)} model(s), proba {enc.shape}", flush=True)
    proba = enc
    if args.llm_npz:
        lp = np.load(args.llm_npz, allow_pickle=True)["proba"]
        assert len(lp) == len(df), f"LLM npz len {len(lp)} != test {len(df)}"
        proba = (1 - args.w_llm) * enc + args.w_llm * lp
        print(f"[ens] +LLM w={args.w_llm}", flush=True)

    pred_ids = proba.argmax(1)
    labels = [S.ID2LABEL[i] for i in pred_ids]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_zip)), exist_ok=True)
    txt_path = os.path.join(os.path.dirname(os.path.abspath(args.out_zip)), args.txt_name)
    open(txt_path, "w").write("\n".join(labels) + "\n")
    with zipfile.ZipFile(args.out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(txt_path, args.txt_name)
    dist = {S.ID2LABEL[k]: int((pred_ids == k).sum()) for k in range(3)}
    print(f"[out] {len(labels)} preds -> {args.out_zip} (flat). dist={dist}", flush=True)

    if "stance" in df.columns and (df["stance"].str.strip() != "").all():
        y = np.array([S.LABEL2ID[s.strip()] for s in df["stance"]])
        m = S.compute_metrics(y, pred_ids)
        print(f"[VALIDATION vs gold] Favg2={m['Favg2']*100:.2f} Favg3={m['Favg3']*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
