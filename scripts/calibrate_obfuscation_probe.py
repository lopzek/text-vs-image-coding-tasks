#!/usr/bin/env python3
"""Memorisation probe of the legibility calibration (obfuscated-identifier control).

Gemma-4's transcription accuracy stayed ~0.82 down to 5-px glyphs where
Qwen3.8 and RapidOCR collapse, and its outputs reproduce public-repo code the
page could not have conveyed — recall, not reading. The knee rule is blind to
this. This probe re-renders a subset of the SAME calibration pages with
every identifier consistently replaced by a same-length pseudo-word (layout,
geometry and ladder byte-for-byte the harness's), transcribes them with the
same VLM path, and compares the accuracy ladder against the original pages.
If the obfuscated ladder collapses where the original stays flat, the
original calibration is memorisation-contaminated.

Subcommands: ``gen`` (CPU) → ``transcribe --model M`` (GPU, same engine as
calibrate_legibility.py) → ``analyze``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import calibrate_legibility as G  # noqa: E402

STOP = set(
    """
    and or not in is if else elif for while def class return import from as with
    try except finally raise yield lambda pass break continue global nonlocal del
    assert async await True False None self cls print len str int float list dict
    set tuple range open file the of to a an by on at be it this that for with
    public private protected static final void int long double float boolean char
    byte short new return null true false class interface extends implements
    throws throw try catch finally package import abstract override var let const
    function string bool using namespace foreach object
    """.split()
)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_VOWELS = "aeiou"
_CONS = "bcdfghjklmnprstvwxz"


def _pseudo_word(length, rng):
    out = []
    for i in range(length):
        pool = _CONS if i % 2 == 0 else _VOWELS
        out.append(rng.choice(pool))
    return "".join(out)


def obfuscate(text, seed):
    """Consistently rename every identifier-like token (len ≥ 3, not in STOP)
    to a same-length pseudo-word, preserving case pattern, leading
    underscores and digits; whitespace/punctuation untouched."""
    rng = random.Random(seed)
    mapping = {}

    def rename(tok):
        if len(tok) < 3 or tok in STOP or tok.lower() in STOP:
            return tok
        if tok in mapping:
            return mapping[tok]
        lead = len(tok) - len(tok.lstrip("_"))
        core = tok[lead:]
        letters = [c for c in core if c.isalpha()]
        pw = _pseudo_word(len(letters), rng)
        it = iter(pw)
        new = []
        for c in core:
            if c.isalpha():
                n = next(it)
                new.append(n.upper() if c.isupper() else n)
            else:
                new.append(c)
        mapping[tok] = "_" * lead + "".join(new)
        return mapping[tok]

    return _IDENT.sub(lambda m: rename(m.group(0)), text), mapping


def select_pages(manifest_rows, every=4):
    pages = {}
    for r in manifest_rows:
        if r["style"] == "plain" and float(r["ladder_ratio"]) == 1.0:
            pages[r["page_id"]] = r
    ids = sorted(pages)
    return [pages[p] for i, p in enumerate(ids) if i % every == 0]


def gen(args):
    rows = [json.loads(line) for line in open(Path(args.in_dir) / "manifest.jsonl")]
    chosen = select_pages(rows, args.every)
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    metric = G.ProcessorCharPxMetric()
    out_rows = []
    for src in chosen:
        seed = int(hashlib.sha256(src["page_id"].encode()).hexdigest(), 16) % (2**31)
        obf, mapping = obfuscate(src["ground_truth"], seed)
        assert len(obf.splitlines()) == len(src["ground_truth"].splitlines())
        variants = G.build_page_variants(
            obf,
            styles=("plain",),
            ladder=tuple(float(x) for x in args.ladder.split(",")),
            width=int(src["native_size"][0]) if False else args.width,
            height=args.height,
            font_size=int(src["font_size"]),
            language=src["language"],
            metric_fn=metric,
        )
        for v in variants:
            image = v.pop("_image")
            fn = f"{src['page_id']}__obf__r{G._ratio_slug(v['ladder_ratio'])}.png"
            image.save(out_dir / "images" / fn, format="PNG")
            out_rows.append(
                {
                    **v,
                    "page_id": src["page_id"],
                    "task": src["task"],
                    "language": src["language"],
                    "original_excerpt_sha256": src["excerpt_sha256"],
                    "obfuscated_sha256": hashlib.sha256(obf.encode()).hexdigest(),
                    "n_identifiers_renamed": len(mapping),
                    "image_path": f"images/{fn}",
                }
            )
    G.write_manifest(out_rows, out_dir / "manifest.jsonl")
    print(f"obfuscation probe gen: pages={len(chosen)} rows={len(out_rows)} out={out_dir}")
    return 0


def transcribe(args):
    from PIL import Image

    in_dir = Path(args.in_dir)
    rows = [json.loads(line) for line in open(in_dir / "manifest.jsonl")]
    vlm = G._make_vlm(
        args.model,
        "vllm",
        args.tensor_parallel_size,
        enforce_eager=True,
        max_num_seqs=args.max_num_seqs,
    )
    out = []
    for r in rows:
        with Image.open(in_dir / r["image_path"]) as im:
            pred = vlm.generate(
                [im.convert("RGB")],
                G.TRANSCRIPTION_PROMPT,
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
        out.append(
            {
                "page_id": r["page_id"],
                "task": r["task"],
                "ladder_ratio": r["ladder_ratio"],
                "model": args.model,
                "model_revision": G._encoder_revision(args.model),
                "char_px_effective": r["char_px_effective_by_encoder"][args.model],
                "char_accuracy": G.char_accuracy(pred, r["ground_truth"]),
                "char_accuracy_fence_stripped": G.char_accuracy(
                    G.strip_markdown_fences(pred), r["ground_truth"]
                ),
                "transcription": pred,
                "ground_truth": r["ground_truth"],
                "obfuscated": True,
            }
        )
        G.write_manifest(out, args.out)
    print(f"obfuscation probe transcribe: model={args.model} rows={len(out)} out={args.out}")
    return 0


def ladder_means(rows, key="char_accuracy"):
    g = defaultdict(list)
    for r in rows:
        g[float(r["ladder_ratio"])].append(float(r[key]))
    return {k: sum(v) / len(v) for k, v in sorted(g.items())}


def analyze(args):
    obf = [json.loads(line) for line in open(args.obfuscated)]
    orig = [json.loads(line) for line in open(args.original)]
    pages = {r["page_id"] for r in obf}
    model = obf[0]["model"]
    orig_same = [
        r for r in orig if r["page_id"] in pages and r["style"] == "plain" and r["model"] == model
    ]
    rep = {
        "model": model,
        "n_pages": len(pages),
        "original_same_pages": ladder_means(orig_same),
        "obfuscated": ladder_means(obf),
        "original_fence_stripped": ladder_means(orig_same, "char_accuracy_fence_stripped"),
        "obfuscated_fence_stripped": ladder_means(obf, "char_accuracy_fence_stripped"),
    }
    rep["drop_at_lowest_rung"] = (
        rep["original_same_pages"][min(rep["original_same_pages"])]
        - rep["obfuscated"][min(rep["obfuscated"])]
    )
    Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    print(f"{model}: pages={len(pages)}")
    for r in sorted(rep["obfuscated"]):
        print(
            f"  r={r:<5} original={rep['original_same_pages'].get(r, float('nan')):.3f}  obfuscated={rep['obfuscated'][r]:.3f}"
        )
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--in-dir", required=True, help="calibration pages dir (manifest.jsonl)")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--every", type=int, default=4)
    g.add_argument("--ladder", default="1.0,0.75,0.5,0.35,0.25")
    g.add_argument("--width", type=int, default=G.DEFAULT_WIDTH)
    g.add_argument("--height", type=int, default=G.DEFAULT_HEIGHT)
    g.set_defaults(fn=gen)
    t = sub.add_parser("transcribe")
    t.add_argument("--model", required=True)
    t.add_argument("--in-dir", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--tensor-parallel-size", type=int, default=4)
    t.add_argument("--max-num-seqs", type=int, default=8)
    t.add_argument("--max-tokens", type=int, default=4096)
    t.set_defaults(fn=transcribe)
    a = sub.add_parser("analyze")
    a.add_argument("--obfuscated", required=True)
    a.add_argument("--original", required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(fn=analyze)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
