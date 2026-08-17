"""
gen_synth.py — Generate synthetic Arabic stance tweets for NEW targets (deepseek, LOGIN node).

Adds target DIVERSITY to encoder training so it learns target-agnostic stance cues → better
unseen-target generalization (Track 2). New targets are distinct from the 3 seen train targets.
Runs in chunks (pass a few --targets per call) to finish within an active turn.

Output CSV columns: ID,text,target,stance (the columns train.py/stance_lib need; aux absent).
"""
import os, sys, re, csv, argparse, time
REPO = os.environ.get("LLM_REGISTRY_ROOT", ".")  # dir holding llm_provider_registry.py + .env
sys.path.insert(0, REPO)
import providers
from pathlib import Path
providers.load_dotenv(Path(REPO) / ".env")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
import llm_predict as L

_, CFG = providers.provider_for("deepseek-v4-flash")
STANCE_DEF = {
    "Favor": "clearly SUPPORTS, endorses, or is glad about",
    "Against": "clearly OPPOSES, criticizes, distrusts, fears, or mocks",
    "None": "mentions but takes NO clear stance on (a neutral question, factual/news statement, or off-topic mention of)",
}


def gen(target, stance, k):
    prompt = (
        f"Generate exactly {k} diverse, realistic Arabic tweets (Gulf/Saudi dialectal or MSA, X/Twitter style — "
        f"may use hashtags, emojis, dialect, abbreviations) where the AUTHOR {STANCE_DEF[stance]} the topic: \"{target}\".\n"
        f"Vary length, tone, and vocabulary so they look like real different users. Each tweet on ONE line; "
        f"no numbering, no quotes, no English, no translations. Output ONLY the {k} Arabic tweets."
    )
    msgs = [{"role": "user", "content": prompt}]
    content = L.call_api(CFG, "deepseek-v4-flash", msgs, "none", 70 * k + 300, temperature=0.8)
    out = []
    for ln in content.splitlines():
        s = re.sub(r"^\s*[\d\-\.\)•\*]+\s*", "", ln).strip()
        if len(s) >= 15 and re.search(r"[؀-ۿ]", s):  # has Arabic
            out.append(re.sub(r"\s+", " ", s))
    return out[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="semicolon-separated English target names")
    ap.add_argument("--per_stance", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "_llm", "synth.csv"))
    args = ap.parse_args()
    targets = [t.strip() for t in args.targets.split(";") if t.strip()]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new_file = not os.path.exists(args.out)
    n = 0
    t0 = time.time()
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ID", "text", "target", "stance"])
        for tgt in targets:
            for st in ["Favor", "Against", "None"]:
                tweets = gen(tgt, st, args.per_stance)
                for i, tw in enumerate(tweets):
                    w.writerow([f"syn_{tgt.replace(' ','_')}_{st}_{i}", tw, tgt, st])
                n += len(tweets)
                print(f"  {tgt} / {st}: {len(tweets)}", flush=True)
    print(f"[RESULT] wrote {n} synthetic tweets ({len(targets)} targets) -> {args.out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
