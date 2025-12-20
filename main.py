import argparse
import json
import os
from typing import Dict, Any

from reasoner.data import load_examples, load_token_descriptions, write_jsonl
from reasoner.tokenizer import ClosedVocabTokenizer
from reasoner.train import train_reasoner, train_dual, train_triple
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
    reasoner, selector, metrics = train_dual(
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


def cmd_train_triple(args):
    ex = load_examples(args.data)
    texts = [t for _, t in ex]
    token_desc = load_token_descriptions(args.token_desc) if args.token_desc else {}
    # Simplified: train only reasoner + selector (no critic/controller needed)
    reasoner, selector, metrics = train_triple(
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
    print(f"[train-triple] reasoner vocab={len(reasoner.vocab)} dim={reasoner.vectors.shape[1]} saved to {args.artifacts}")
    print(f"[train-triple] selector contexts={len(selector.trigram_logp)} (trigram) + {len(selector.bigram_logp)} (bigram) saved to {args.artifacts}")


# Removed: cmd_train_controller - no longer needed (using rule-based quality scoring)


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


def cmd_generate_triple(args):
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
    p_tr.add_argument("--dim", type=int, default=128)
    p_tr.add_argument("--window", type=int, default=5)
    p_tr.add_argument("--desc_alpha", type=float, default=0.40)
    p_tr.set_defaults(func=cmd_train)

    p_td = sub.add_parser("train-dual")
    p_td.add_argument("--data", required=True)
    p_td.add_argument("--token_desc", required=False, default="")
    p_td.add_argument("--artifacts", required=True)
    p_td.add_argument("--dim", type=int, default=128)
    p_td.add_argument("--window", type=int, default=5)
    p_td.add_argument("--desc_alpha", type=float, default=0.40)
    p_td.add_argument("--selector_smooth", type=float, default=0.5)
    p_td.add_argument("--selector_max_per_context", type=int, default=256)
    p_td.set_defaults(func=cmd_train_dual)

    p_tt = sub.add_parser("train-triple")
    p_tt.add_argument("--data", required=True)
    p_tt.add_argument("--token_desc", required=False, default="")
    p_tt.add_argument("--artifacts", required=True)
    p_tt.add_argument("--dim", type=int, default=128)
    p_tt.add_argument("--window", type=int, default=5)
    p_tt.add_argument("--desc_alpha", type=float, default=0.40)
    p_tt.add_argument("--selector_smooth", type=float, default=0.5)
    p_tt.add_argument("--selector_max_per_context", type=int, default=256)
    # Note: critic/controller removed - using rule-based quality scoring instead
    p_tt.set_defaults(func=cmd_train_triple)

# Removed: train-controller command - no longer needed

    p_ge = sub.add_parser("generate")
    p_ge.add_argument("--artifacts", required=True)
    p_ge.add_argument("--prompt", required=True)
    p_ge.add_argument("--max_new_tokens", type=int, default=30)
    p_ge.add_argument("--temperature", type=float, default=0.70)
    p_ge.add_argument("--top_k", type=int, default=12)
    p_ge.add_argument("--explain", type=int, default=0)
    p_ge.add_argument("--block_dataset_meta", type=int, default=1)
    p_ge.add_argument("--repeat_window", type=int, default=12)
    p_ge.add_argument("--repetition_penalty", type=float, default=2.25)
    p_ge.add_argument("--semantic_repeat_window", type=int, default=6)
    p_ge.add_argument("--semantic_repeat_threshold", type=float, default=0.68)
    p_ge.add_argument("--semantic_repeat_penalty", type=float, default=1.15)
    p_ge.set_defaults(func=cmd_generate)

    p_gd = sub.add_parser("generate-dual")
    p_gd.add_argument("--artifacts", required=True)
    p_gd.add_argument("--prompt", required=True)
    p_gd.add_argument("--max_new_tokens", type=int, default=60)
    p_gd.add_argument("--temperature", type=float, default=0.70)
    p_gd.add_argument("--selector_top_k", type=int, default=40)
    p_gd.add_argument("--alpha_selector", type=float, default=0.45)
    p_gd.add_argument("--beta_reasoner", type=float, default=0.55)
    p_gd.add_argument("--eta_trust", type=float, default=0.10)
    p_gd.add_argument("--eta_bias", type=float, default=0.08)
    p_gd.add_argument("--trust_clip", type=float, default=2.0)
    p_gd.add_argument("--w_semantic", type=float, default=1.25)
    p_gd.add_argument("--w_anchor", type=float, default=0.85)
    p_gd.add_argument("--w_contra", type=float, default=1.45)
    p_gd.add_argument("--w_repeat", type=float, default=1.35)
    p_gd.add_argument("--repeat_window", type=int, default=6)
    p_gd.add_argument("--context_window", type=int, default=10)
    p_gd.add_argument("--block_dataset_meta", type=int, default=1)
    p_gd.add_argument("--explain", type=int, default=0)

    p_gd.add_argument("--lookahead_len", type=int, default=10)
    p_gd.add_argument("--num_continuations", type=int, default=18)
    p_gd.add_argument("--commit_len", type=int, default=1)
    p_gd.add_argument("--w_meta_frame", type=float, default=0.45)

    p_gd.set_defaults(func=cmd_generate_dual)

    p_gt = sub.add_parser("generate-triple")
    p_gt.add_argument("--artifacts", required=True)
    p_gt.add_argument("--prompt", required=True)
    p_gt.add_argument("--max_new_tokens", type=int, default=60)
    p_gt.add_argument("--temperature", type=float, default=0.85)
    p_gt.add_argument("--selector_top_k", type=int, default=32)
    p_gt.add_argument("--alpha_selector", type=float, default=0.55)
    p_gt.add_argument("--beta_reasoner", type=float, default=0.45)
    p_gt.add_argument("--eta_trust", type=float, default=0.10)
    p_gt.add_argument("--eta_bias", type=float, default=0.08)
    p_gt.add_argument("--trust_clip", type=float, default=2.0)
    p_gt.add_argument("--w_semantic", type=float, default=1.0)
    p_gt.add_argument("--w_anchor", type=float, default=0.55)
    p_gt.add_argument("--w_contra", type=float, default=1.25)
    p_gt.add_argument("--w_repeat", type=float, default=0.85)
    p_gt.add_argument("--repeat_window", type=int, default=6)
    p_gt.add_argument("--context_window", type=int, default=10)
    p_gt.add_argument("--block_dataset_meta", type=int, default=1)
    p_gt.add_argument("--explain", type=int, default=0)
    p_gt.add_argument("--lookahead_len", type=int, default=6)
    p_gt.add_argument("--num_continuations", type=int, default=12)
    p_gt.add_argument("--commit_len", type=int, default=1)
    p_gt.add_argument("--w_meta_frame", type=float, default=0.45)
    p_gt.set_defaults(func=cmd_generate_triple)

    p_v = sub.add_parser("vocab")
    p_v.add_argument("--data", required=True)
    p_v.set_defaults(func=cmd_vocab)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
