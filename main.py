import argparse
import json

from reasoner.data import load_examples, load_token_descriptions, write_jsonl
from reasoner.tokenizer import ClosedVocabTokenizer
from reasoner.train import train_reasoner, train_dual
from reasoner.model import ReasonerArtifacts
from reasoner.selector import SelectorArtifacts
from reasoner.generate import generate, generate_dual


def cmd_build_token_desc(args):
    ex = load_examples(args.data)
    texts = [t for _, t in ex]
    tok = ClosedVocabTokenizer.from_texts(texts)
    rows = [{"token": t, "description": ""} for t in tok.vocab]
    write_jsonl(args.out, rows)
    print(f"[build-token-descriptions] wrote {len(rows)} tokens to {args.out}")


def cmd_train(args):
    ex = load_examples(args.data)
    texts = [t for _, t in ex]
    token_desc = load_token_descriptions(args.token_desc) if args.token_desc else {}
    art = train_reasoner(
        texts=texts,
        token_desc=token_desc,
        dim=args.dim,
        window=args.window,
        desc_alpha=args.desc_alpha,
    )
    art.save(args.artifacts)
    print(f"[train] vocab={len(art.vocab)} dim={art.vectors.shape[1]} saved to {args.artifacts}")


def cmd_train_dual(args):
    ex = load_examples(args.data)
    texts = [t for _, t in ex]
    token_desc = load_token_descriptions(args.token_desc) if args.token_desc else {}
    reasoner, selector = train_dual(
        texts=texts,
        token_desc=token_desc,
        dim=args.dim,
        window=args.window,
        desc_alpha=args.desc_alpha,
        selector_smooth=args.selector_smooth,
        selector_max_per_context=args.selector_max_per_context,
    )
    reasoner.save(args.artifacts)
    selector.save(args.artifacts)
    print(f"[train-dual] reasoner vocab={len(reasoner.vocab)} dim={reasoner.vectors.shape[1]} saved to {args.artifacts}")
    print(f"[train-dual] selector contexts={len(selector.trigram_logp)} (trigram) + {len(selector.bigram_logp)} (bigram) saved to {args.artifacts}")


def cmd_generate(args):
    art = ReasonerArtifacts.load(args.artifacts)
    out = generate(
        art=art,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        explain=bool(args.explain),
        block_dataset_meta=bool(args.block_dataset_meta),
        repeat_window=args.repeat_window,
        repetition_penalty=args.repetition_penalty,
        semantic_repeat_window=args.semantic_repeat_window,
        semantic_repeat_threshold=args.semantic_repeat_threshold,
        semantic_repeat_penalty=args.semantic_repeat_penalty,
        style=args.style,
        min_sentence_tokens=args.min_sentence_tokens,
        max_sentence_tokens=args.max_sentence_tokens,
        closure_strength=args.closure_strength,
        cluster_switch_window=args.cluster_switch_window,
        cluster_switch_penalty=args.cluster_switch_penalty,
    )
    print(out["text"])
    if args.explain:
        print("\n--- EXPLANATION (top candidates per step) ---")
        print(json.dumps(out["explain"], ensure_ascii=False, indent=2))


def cmd_generate_dual(args):
    reasoner = ReasonerArtifacts.load(args.artifacts)
    selector = SelectorArtifacts.load(args.artifacts)
    out = generate_dual(
        reasoner=reasoner,
        selector=selector,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        selector_top_k=args.selector_top_k,
        alpha_selector=args.alpha_selector,
        beta_reasoner=args.beta_reasoner,
        eta_trust=args.eta_trust,
        eta_bias=args.eta_bias,
        trust_clip=args.trust_clip,
        w_semantic=args.w_semantic,
        w_anchor=args.w_anchor,
        w_contra=args.w_contra,
        w_repeat=args.w_repeat,
        repeat_window=args.repeat_window,
        block_dataset_meta=bool(args.block_dataset_meta),
        explain=bool(args.explain),
        context_window=args.context_window,
        # micro-continuation debate knobs:
        lookahead_len=args.lookahead_len,
        num_continuations=args.num_continuations,
        commit_len=args.commit_len,
        w_meta_frame=args.w_meta_frame,
    )
    print(out["text"])
    if args.explain:
        print("\n--- DEBATE TRACE (top continuations per step) ---")
        print(json.dumps(out.get("explain", []), ensure_ascii=False, indent=2))
        print("\n--- ONLINE STATE ---")
        print(json.dumps(out.get("online", {}), ensure_ascii=False, indent=2))


def cmd_finetune(args):
    try:
        from reasoner.finetune.data import load_io_jsonl
        from reasoner.finetune.finetune import finetune_reasoner
    except ImportError:
        raise ImportError("Finetune module not found. Please ensure reasoner/finetune/data.py and reasoner/finetune/finetune.py exist.")
    base = ReasonerArtifacts.load(args.base_artifacts)
    pairs = load_io_jsonl(args.data, input_key=args.input_key, output_key=args.output_key)
    token_desc = load_token_descriptions(args.token_desc) if args.token_desc else {}
    art = finetune_reasoner(
        base=base,
        io_pairs=pairs,
        token_desc=token_desc,
        window=args.window,
        mix_bigram=args.mix_bigram,
        mix_vectors=args.mix_vectors,
        corruption_swaps=args.corruption_swaps,
        corruption_quantile=args.corruption_quantile,
        seed=args.seed,
    )
    art.save(args.out_artifacts)
    print(f"[finetune] wrote updated artifacts to: {args.out_artifacts} (pairs={len(pairs)})")


def cmd_vocab(args):
    ex = load_examples(args.data)
    texts = [t for _, t in ex]
    tok = ClosedVocabTokenizer.from_texts(texts)
    print(f"vocab size: {len(tok.vocab)}")
    print(tok.vocab)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_bt = sub.add_parser("build-token-descriptions")
    p_bt.add_argument("--data", required=True)
    p_bt.add_argument("--out", required=True)
    p_bt.set_defaults(func=cmd_build_token_desc)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--data", required=True)
    p_tr.add_argument("--token_desc", required=False, default="")
    p_tr.add_argument("--artifacts", required=True)
    p_tr.add_argument("--dim", type=int, default=32)
    p_tr.add_argument("--window", type=int, default=2)
    p_tr.add_argument("--desc_alpha", type=float, default=0.35)
    p_tr.set_defaults(func=cmd_train)

    p_td = sub.add_parser("train-dual")
    p_td.add_argument("--data", required=True)
    p_td.add_argument("--token_desc", required=False, default="")
    p_td.add_argument("--artifacts", required=True)
    p_td.add_argument("--dim", type=int, default=32)
    p_td.add_argument("--window", type=int, default=2)
    p_td.add_argument("--desc_alpha", type=float, default=0.35)
    p_td.add_argument("--selector_smooth", type=float, default=0.5)
    p_td.add_argument("--selector_max_per_context", type=int, default=256)
    p_td.set_defaults(func=cmd_train_dual)

    p_ft = sub.add_parser("finetune")
    p_ft.add_argument("--base_artifacts", required=True, help="Path to existing artifacts directory (pretraining)")
    p_ft.add_argument("--data", required=True, help="JSONL with conversation input/output pairs")
    p_ft.add_argument("--input_key", default="input", help="JSON key for the input field")
    p_ft.add_argument("--output_key", default="output", help="JSON key for the output field")
    p_ft.add_argument("--token_desc", required=False, default="", help="Optional token_descriptions.jsonl for manual descriptions")
    p_ft.add_argument("--out_artifacts", required=True, help="Output artifacts directory for finetuned model")
    p_ft.add_argument("--window", type=int, default=2, help="Co-occurrence window used for PPMI vectors and cooc stats")
    p_ft.add_argument("--mix_bigram", type=float, default=0.35, help="Mix ratio for updating bigram transitions (0..1)")
    p_ft.add_argument("--mix_vectors", type=float, default=0.35, help="Mix ratio for updating semantic vectors (0..1)")
    p_ft.add_argument("--corruption_swaps", type=int, default=2, help="Strength of structural corruption used to derive incoherence signals")
    p_ft.add_argument("--corruption_quantile", type=float, default=0.995, help="Quantile for selecting corrupted-heavy transitions (0.5..0.9999)")
    p_ft.add_argument("--seed", type=int, default=7)
    p_ft.set_defaults(func=cmd_finetune)

    p_ge = sub.add_parser("generate")
    p_ge.add_argument("--artifacts", required=True)
    p_ge.add_argument("--prompt", required=True)
    p_ge.add_argument("--max_new_tokens", type=int, default=30)
    p_ge.add_argument("--temperature", type=float, default=0.8)
    p_ge.add_argument("--top_k", type=int, default=8)
    p_ge.add_argument("--explain", type=int, default=0)
    p_ge.add_argument("--block_dataset_meta", type=int, default=1, help="Block dataset/meta tokens from output (1=yes, 0=no)")
    p_ge.add_argument("--repeat_window", type=int, default=3, help="Window size for exact repetition penalty")
    p_ge.add_argument("--repetition_penalty", type=float, default=1.25, help="Penalty strength for exact token repetition")
    p_ge.add_argument("--semantic_repeat_window", type=int, default=2, help="Window size for semantic repetition penalty")
    p_ge.add_argument("--semantic_repeat_threshold", type=float, default=0.72, help="Cosine threshold above which semantic repetition is penalized")
    p_ge.add_argument("--semantic_repeat_penalty", type=float, default=0.85, help="Penalty strength for semantic repetition")
    p_ge.add_argument("--style", choices=["descriptive", "explanatory"], default="descriptive")
    p_ge.add_argument("--min_sentence_tokens", type=int, default=10)
    p_ge.add_argument("--max_sentence_tokens", type=int, default=26)
    p_ge.add_argument("--closure_strength", type=float, default=1.15)
    p_ge.add_argument("--cluster_switch_window", type=int, default=2)
    p_ge.add_argument("--cluster_switch_penalty", type=float, default=0.28)
    p_ge.set_defaults(func=cmd_generate)

    p_gd = sub.add_parser("generate-dual")
    p_gd.add_argument("--artifacts", required=True)
    p_gd.add_argument("--prompt", required=True)
    p_gd.add_argument("--max_new_tokens", type=int, default=60)
    p_gd.add_argument("--temperature", type=float, default=0.85)
    p_gd.add_argument("--selector_top_k", type=int, default=32)
    p_gd.add_argument("--alpha_selector", type=float, default=0.55)
    p_gd.add_argument("--beta_reasoner", type=float, default=0.45)
    p_gd.add_argument("--eta_trust", type=float, default=0.10)
    p_gd.add_argument("--eta_bias", type=float, default=0.08)
    p_gd.add_argument("--trust_clip", type=float, default=2.0)
    p_gd.add_argument("--w_semantic", type=float, default=1.0)
    p_gd.add_argument("--w_anchor", type=float, default=0.55)
    p_gd.add_argument("--w_contra", type=float, default=1.25)
    p_gd.add_argument("--w_repeat", type=float, default=0.85)
    p_gd.add_argument("--repeat_window", type=int, default=6)
    p_gd.add_argument("--context_window", type=int, default=10)
    p_gd.add_argument("--block_dataset_meta", type=int, default=1)
    p_gd.add_argument("--explain", type=int, default=0)

    # NEW: micro-continuation debate controls
    p_gd.add_argument("--lookahead_len", type=int, default=6, help="Continuation length used for sequence-level debate")
    p_gd.add_argument("--num_continuations", type=int, default=12, help="How many continuations to sample per step")
    p_gd.add_argument("--commit_len", type=int, default=1, help="How many tokens to commit from the chosen continuation per step")
    p_gd.add_argument("--w_meta_frame", type=float, default=0.45, help="Penalty weight for meta framing roles at the continuation level")

    p_gd.set_defaults(func=cmd_generate_dual)

    p_v = sub.add_parser("vocab")
    p_v.add_argument("--data", required=True)
    p_v.set_defaults(func=cmd_vocab)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
